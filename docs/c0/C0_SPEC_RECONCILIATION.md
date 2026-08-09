# C0 — Three-way spec reconciliation

**ORIGINAL VERSION-B SPEC** vs **FROZEN VERSION-B REALITY** vs **VERSION-C v1.1 REQUIREMENT**

Every row carries a Version-B status and a Version-C action. Nothing here is written from
memory: the Version-B reality column is read from the frozen repository at
`7799f7decd35db6987ce4578824e5bd8d9eab4ae`, and the requirement columns are read from the
DOCX files fingerprinted in §1.

## 1. Documents used, and the ambiguity in "the original Version-B spec"

| Role | Path | SHA256 | Size |
|---|---|---|---|
| **Version-C authority** | `D:\AI on IOT\Anti_spoofing\PRISM_FAS_C_LLM_v1_1_FINAL_Detailed_Spec_2026.docx` | `d9edbfefa6829f29bb075e2f3d12073bb6517be57c0debcf93c92c4346d2e2df` | 455 241 B |
| **Version-B original design (v1.0)** | `D:\AI on IOT\Anti_spoofing\References\Spec_PRISM_FAS_Version_B_Detailed_2026.docx` | `74c7557f14712b7c5fcc347a86b5f9abc03c6225155bdca275aea38798ff3292` | 830 417 B |
| **Version-B v1.1 implementation contract** | `D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_v1_1_CASIA_MSU_SiWMv2_LocalPreprocess_DualBackend_Codex_Spec.docx` | `44634d7cdcddc0bd0ed54e78407d62be60de275f45461024573dd3f94d0cfeac` | 417 956 B |

All three are copied verbatim into [`docs/spec/`](../spec/). The originals on disk were
not altered. (The v1.1 DOCX was open in Word during C0 and was read through a shared-read
handle; its SHA256 matches the one Version B itself recorded in `docs/SPEC_SOURCE.json`,
which is independent confirmation that the byte content is the one B implemented.)

### 1.1 Documented ambiguity — resolved, not silently chosen

The instruction for C0 expected an original B spec "similar to
`PRISM_FAS_B_v1_1_CASIA_MSU_SiWMv2_...`". Two documents match the family, and a third
near-miss exists. They are **not** in conflict; they are different layers, and the
Version-C spec itself names both:

- Version-C spec §"Nguồn tài liệu dùng để xây spec này" cites
  `Spec_PRISM_FAS_Version_B_Detailed_2026(2).docx` as *"đặc tả gốc có LLM recipe engine,
  ontology 5 trục, GPAT và regional manifold"* — the **original design intent**. The file
  on this machine is `References\Spec_PRISM_FAS_Version_B_Detailed_2026.docx`, without the
  `(2)` download suffix. Content matches the description exactly (§1.1 pins a Recipe LLM;
  §6 is "Attack ontology và LLM recipe engine"), so it is treated as that document.
- The same section cites `PRISM_FAS_B_v1_1_CASIA_MSU_SiWMv2_LocalPreprocess_DualBackend_Codex_Spec.docx`
  as the v1.1 document that *"đã chốt source CASIA+MSU, target SiW-Mv2, recipe schema v1.1
  và MVP physics+GPAT"* — the **implementation contract**. Version B's own
  `docs/SPEC_SOURCE.json` records precisely this file and this SHA256 as the spec it was
  built against.
- `References\Spec_PRISM_FAS_Version_C_Detailed_2026.docx`
  (SHA256 `14f5ee351b0612201061b68d455aa204ee1196a8f4c64dc6383185f0e063963d`) is an
  **earlier, superseded Version-C draft**. It is NOT the authority for this project. The
  authority is the `..._v1_1_FINAL_...` file in the work root. This draft was read only to
  confirm it is superseded and is deliberately not copied into `docs/spec/`.

So "the original Version-B spec" resolves to a **pair**: v1.0 for design intent, v1.1 for
the implementation contract. Where they disagree, v1.1 governs what B was obliged to do,
and the disagreement itself is recorded as a row below.

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| `EXACT` | Version B implemented the requirement as specified. |
| `PARTIAL` | Implemented, but not to the full declared extent. |
| `DEVIATED` | Implemented differently from the specified value or mechanism. |
| `OMITTED` | Specified but not implemented. |
| `SUPERSEDED_WITH_JUSTIFICATION` | Deliberately replaced, with the reason recorded in B. |
| `NOT_ESTABLISHED` | Neither spec settled it; B's behaviour was an unexamined default. |

## 3. Compliance matrix

Requirements audited: **37**.

| # | Requirement | Original B spec intent | Frozen B reality | B status | Version-C v1.1 requirement | Version-C action |
|---|---|---|---|---|---|---|
| 1 | Dataset roles | v1.0 §5.1: CASIA, MSU, **CelebA-Spoof**, SiW-Mv2 | CASIA + MSU source; SiW-Mv2 target-only. CelebA-Spoof absent | `SUPERSEDED_WITH_JUSTIFICATION` | Exactly 3: CASIA, MSU, SiW-Mv2; no fourth dataset in v1.1 | Keep. Freeze the 3-dataset list in config; a fourth dataset requires protocol v2 |
| 2 | Cross-domain protocols | v1.0 §5.3: P-B0…P-B3 + leave-one-attack-out | One direction only: CASIA+MSU → SiW-Mv2 | `PARTIAL` | Three protocols P1 CASIA→MSU, P2 MSU→CASIA, P3 CASIA+MSU→SiW-Mv2 | **Add P1 and P2.** Build subject/video-disjoint manifests for all three at C0/C1 |
| 3 | Source split policy | v1.1 §3.2: official train→`source_train`, official test→`source_dev`; no frame-level split | Implemented exactly; all frames of a video stay in one split | `EXACT` | Same, per protocol, with leakage = 0 proven before training | Inherit. Re-prove leakage = 0 per protocol manifest |
| 4 | Preprocessing — detector and crop | v1.1 §4.5: SCRFD-**2.5G**, crop **256×256 lossless PNG**, "no re-encode JPEG" | SCRFD-**10G** (`scrfd_10g_bnkps`), input 320, threshold 0.50, padding 0.25, crop **224×224 JPEG q95** | `DEVIATED` | §5.2 codifies the *actual* contract: SCRFD input 320, threshold 0.5, largest valid face, canonical crop 224×224 JPEG q95 | Inherit B reality unchanged for byte-level package reuse. Version-C spec has already ratified the deviation; do not "restore" the 256-PNG text |
| 5 | Frame sampling density | v1.1 §4.4: 32 train / 32 dev / 48 target frames per video | **4** frames per video (`frames_per_video: 4`); 1700 target videos → 6800 planned, 6776 kept | `DEVIATED` | Not re-specified; the frozen package identity is inherited | Inherit as-is. Record the true density in the C reproducibility appendix; do **not** silently present 32/48 |
| 6 | LLM usage | v1.0 §1.1/§6.2: frozen offline LLM (Qwen2.5-7B / Qwen3-8B) generates JSON recipes | **No external LLM invoked.** `external_llm_invoked: false`, enforced by validator and tests | `OMITTED` | §7: a real Gemini planner is mandatory for the LLM arm | **The core Version-C change.** See [`C0_LLM_GAP_ANALYSIS.md`](C0_LLM_GAP_ANALYSIS.md) |
| 7 | Recipe generation | v1.0 §6.2: LLM candidates → rule validation → diversity filter → freeze | `deterministic-source-only-recipe-generator` m7-v1, 128 recipes, seed 20260806, offline TF-IDF diversity | `DEVIATED` | Three arms RND/DET/LLM, **256 recipes each**; LLM from 384 raw slots | Reimplement DET from the B generator; add RND; add the Gemini LLM arm. Freeze all three at C3 |
| 8 | Recipe schema | v1.1 §7.2: schema v1.1 (medium/geometry/regions/artifacts/capture/route/seed) | Implemented exactly; `schema.py` + canonical hashing | `EXACT` | §7.3: reuse schema v1.1 for comparability; any extension needs a version bump | Inherit unchanged. No schema change in v1.1 |
| 9 | Attack ontology (5 axes) | v1.0 §6, v1.1 §7.4: medium, geometry, region, artifact, capture, severity | `configs/recipes/ontology_m7.yaml`, `m7-ontology-v1`, all axes present | `EXACT` | §7.4 same minimum vocabulary; all 8 artifacts and all 9 regions must appear in each final bank | Inherit the ontology; add the per-bank coverage minimums as a hard selector constraint |
| 10 | Recipe compiler | v1.0 §6.2: JSON → operator graph + mask policy + conditioning vector | `m7-compiler-v1`; deterministic graph hash; unknown field rejected | `EXACT` | §9: same compiler boundary for all three arms | **Inherit and hold the boundary fixed.** It is what makes the generator the only changed factor |
| 11 | Conditioning vector | Required, dimension unspecified | Fixed 41-D `recipe_conditioning_v1` | `PARTIAL` | §9.1: inherit the 41-D encoding if schema-compatible | Inherit 41-D. Verify schema compatibility at C4 before GPAT-C training |
| 12 | Physics route | v1.0 §7.2, v1.1 §8.1: deterministic operators, exact mask, artifact map | 8 operators; exact changed-pixel mask; `max\|out−in\|` outside mask = 0 | `EXACT` | §10.1 mandatory and deterministic; same operator families | Inherit unchanged |
| 13 | GPAT route | v1.0 §7.3: geometry-preserving residual, LL band locked (γ=0), artifact map | 910 538-param Haar wavelet residual generator; LL band structurally absent | `EXACT` | §10.2 same artifact-only residual contract | Inherit the mechanism |
| 14 | GPAT conditioning fairness | Not addressed by either B document | A02 random-recipe control ran against a GPAT trained on a **structured** recipe identity → OOD-conditioning caveat | `NOT_ESTABLISHED` | §8.3: GPAT-C **must** be generator-neutral; one frozen checkpoint shared by all arms | **New requirement.** Train GPAT-C once on an ontology-derived neutral support bank independent of the RND/DET/LLM treatment banks (C4) |
| 15 | Masked diffusion | v1.0 §7.5: optional extension (SDXL inpainting + ControlNet) | Not implemented; explicitly out of MVP | `OMITTED` | §10.3: out of MVP; may only follow after C0–C13 close | Keep out. Adding it before main results would confound the LLM question |
| 16 | Quality gate | v1.0 §7.6 / v1.1 §8.4: 8 gates (detection, identity, NME, parsing Dice, outside-mask, strength, fingerprint, recipe match) | All 8 implemented as hard gates plus soft `q` | `EXACT` | §11.1 same metric families; one common gate for all arms | Inherit. **Freeze one gate identity across arms**; never relax it for a single arm |
| 17 | Quality-gate calibration | Thresholds calibrated source-only; population unspecified | Calibrated three times (v1 → v2 → v3); v1 and v2 retained, not deleted | `SUPERSEDED_WITH_JUSTIFICATION` | §11 same source-only rule | Inherit the v3 methodology; recalibrate for the C candidate population **before** any arm is scored |
| 18 | Synthetic bank | v1.1 §8.5: versioned, quality-gated bank with rejected manifest | 1120 candidates → 871 accepted / 249 rejected; **routes and counts unequal** (physics 419 / gpat 452) | `PARTIAL` | §11.3: exactly **1024 accepted/arm = 512 Physics + 512 GPAT** from 2048 candidates/arm | **New hard requirement.** Matched-bank builder at C6; if an arm cannot reach 1024 under the common gate, C6 FAILS |
| 19 | `q` weighting | v1.0 §10.4 / v1.1 §8.4: soft weight, not a label | `q` multiplies only the synthetic bracket; never a live/spoof label | `EXACT` | §11.2 identical normative rule | Inherit unchanged |
| 20 | ConvNeXt local branch | v1.0 §1.1: ConvNeXt V2 Atto **or** Tiny | Atto, `timm/convnextv2_atto.fcmae_ft_in1k`, weight SHA `6389c2f5…7ebb` | `EXACT` | §13.2: ConvNeXt V2 **Atto fixed**; no Tiny option in v1.1 | Inherit the exact weight identity. The Atto/Tiny degree of freedom is removed |
| 21 | SigLIP2 global branch | v1.0 §1.1: SigLIP2 Base P16-224, frozen first, LoRA later | Frozen, pinned to revision `75de2d55…2fa2`, 7 file SHA-256s verified in-container. LoRA never enabled | `EXACT` | §13.1/§13.2: same frozen model identity/revision inherited unless C0 proves it unavailable | Inherit the identity and revision; it enters the C run identity. LoRA stays off |
| 22 | Semantic regions | v1.0 §8.1: 9 regions, soft masks, never 9 hard crops | 9 regions in the frozen order; soft prior masks; visibility gating | `EXACT` | §13.2: same 9 regions for Track R; regions OFF for Track G | Inherit for Track R. **Track G runs regions OFF** — a new, deliberately simpler primary track |
| 23 | Region fusion | v1.0 §8.4: cross-attention on local features + global token pooling | Implemented literally | `EXACT` | §13.2 same | Inherit for Track R |
| 24 | PromptHead | v1.0 §8.5: InfoNCE alignment of attacked-region embeddings to recipe text | Implemented against a **cached** 128×768 frozen text artifact; no text encoder at train/inference time | `PARTIAL` | §12.2/§12.3: same offline cache per bank; target-time `p_prompt` **must be `null`/not_applicable**, never `0.0` | Inherit the cache mechanism; **fix the target-time semantics explicitly**. C-H5 tests PromptHead on/off at 3 seeds |
| 25 | Multi-prototype real manifold | v1.0 §9: K prototypes per region, Mahalanobis distance | K=4, diagonal covariance, K-means init on source_train live | `EXACT` | §13.2: manifold **OFF** in the primary Track-R variant; K=4 only as an explicit secondary row | **Do not silently restore it.** B's H2 found multi-prototype significantly *worse*; v1.1 demotes it |
| 26 | Loss contract | v1.0 §10: decomposed loss with named terms | Implemented with explicit active flags | `EXACT` | §14.3: reuse the decomposed loss; any change is a typed variant declared before target | Inherit; read the exact implementation and freeze it at C7 |
| 27 | Batch composition | v1.1 §10.2: 12 live + 12 real spoof + 8 synthetic | Exactly that, CASIA/MSU-balanced on both real partitions | `EXACT` | §14.1 identical; `synthetic=none` must truly omit the bracket | Inherit. Add the "no fake zero samples" check |
| 28 | Calibration and threshold transfer | v1.1 §16.2: temperature + threshold fitted on `source_dev` only | Fitted on `source_dev` only — but G7 v1 thresholded the fused `s_final` with a calibration fitted on `global_logit` alone. Found and fixed **before** any label was opened; v1 retained as superseded | `DEVIATED` | §16: the decision score must be named and consistent between calibration and inference — an explicit hard regression guard derived from this defect | **Inherit the guard as a test.** Add a source-only threshold-transfer audit across CASIA-dev and MSU-dev |
| 29 | Target isolation / firewall | v1.1 §6: train cannot mount target; evaluator cannot write model state | Structural: paths compared after `resolve()`; G8's lack of training capability proved by an AST import audit | `EXACT` | §5.4: same capability boundary — TRAIN/LLM/SYNTHESIS cannot resolve SiW labels; C11 label-free only; C12 scorer-only | Inherit the firewall and **extend it to the LLM prompt path**, which did not exist in B |
| 30 | G7/G8 protocol | v1.1 §6.3: predict first, score later, labels joined afterwards | Implemented; predictions frozen, lockset validated twice, pre-reveal audit 15/15, one-way reveal record | `EXACT` | §19.2: same 9-step procedure, renamed C-G7/C-G8 | Inherit wholesale. Rename stages `C-G*` to avoid collision with the historical G-numbers |
| 31 | Experiment matrix | v1.1 §17: B00–B08 baselines + 10 ablation families | 42 logical rows, 38 executable, 4 blocked with reasons, 0 failed | `PARTIAL` | §18.1: 5 primary rows (C-G-RND/DET/LLM, C-R-DET/LLM) + declared diagnostics | Replace the matrix. Generator arm becomes the primary factor; no experiment-id branching |
| 32 | Replication policy | v1.1 §16.3: ≥3 seeds for main baseline and full method | 3 seeds for spec-mandated and hypothesis-critical rows; 1 seed for diagnostics, reported `single_seed_descriptive` | `PARTIAL` | §18.3: **5 seeds** for P3 Track-G C-H1/C-H2; 3 seeds for P1/P2 and Track-R/PromptHead | Raise the primary rows to 5 seeds, family 20260806–20260810. Keep the refusal of comparisons involving a 1-seed row |
| 33 | Statistical protocol | v1.1 §16.3 Table 58: mean ± std over ≥3 seeds, paired video bootstrap | Paired video bootstrap, 10 000 resamples, 95 % percentile CI, seed 20260810, Holm-Bonferroni, structural resample-plan identity | `EXACT` | §20.3: identical, over the family C-H1/C-H2/C-H3/C-H5, with C-H4 reported separately | Inherit the whole module including the plan-identity mechanism. Only the hypothesis family changes |
| 34 | Shortcut / reliability gates | v1.1 §17.3: probe, corruption, residual-zero, region-shift, map-swap, cross-route | 10 tests run: 6 passed, **2 failed** (synthetic-vs-real probe 0.9375 > 0.75; residual-zero movement 0.022 < 0.10), 2 blocked. Failures reported, not re-tuned | `PARTIAL` | §17: same suite, but the probe becomes a **gate before target** — fail ⇒ C6 fails or redesign | **Promote from report to gate.** Version C must resolve the probe failure before P3, not merely disclose it |
| 35 | Compute backend | v1.1 §12: one trainer core, PC **or** Modal, no `modal` import in the core | Modal L4 (fallback L40S) used; trainer core never imports `modal`. The PC full-training row was **blocked** — no local CUDA | `PARTIAL` | §23.3: backend-neutral over `modal_l4` and `ssh_lab`; `ssh_lab` mandatory before C8; one homogeneous profile for hypothesis-critical rows | **Add the `ssh_lab` backend.** Modal access may expire; see [`C0_COMPUTE_BACKEND_PLAN.md`](C0_COMPUTE_BACKEND_PLAN.md) |
| 36 | Reproducibility / identity system | v1.0 §20 + v1.1 §15: hashes for config, package, bank, checkpoint, environment | Full chain implemented; fail-closed on unknown identity; locks validated twice | `EXACT` | §21: same chain extended with LLM prompt/model/raw-response identities | Inherit; **extend the chain** with the LLM provenance record (§7.7, 15 required fields) |
| 37 | Novelty / claim policy | v1.0 §19.2, v1.1 §Trạng thái: no "first", no unproven novelty | Honoured: 3 of 5 hypotheses reported as negative results; no SOTA or first-method claim | `EXACT` | §28: same discipline plus the mandatory SiW historical disclosure | Inherit. Add the standing rule that P3 is never called a never-before-seen blind target |

## 4. Status counts

| Status | Count |
|---|---|
| `EXACT` | **20** |
| `PARTIAL` | **8** |
| `DEVIATED` | **4** |
| `OMITTED` | **2** |
| `SUPERSEDED_WITH_JUSTIFICATION` | **2** |
| `NOT_ESTABLISHED` | **1** |
| **Total audited** | **37** |

`EXACT` rows: 3, 8, 9, 10, 12, 13, 16, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 33, 36, 37.
`PARTIAL`: 2, 11, 18, 24, 31, 32, 34, 35. `DEVIATED`: 4, 5, 7, 28. `OMITTED`: 6, 15.
`SUPERSEDED_WITH_JUSTIFICATION`: 1, 17. `NOT_ESTABLISHED`: 14.

## 5. What this tells Version C

The inherited infrastructure is in far better shape than the inherited *science*. Twenty
of thirty-seven requirements were implemented exactly, and they are the expensive ones:
preprocessing, compiler, physics, GPAT mechanism, quality gate, firewall, G7/G8 protocol,
statistics, identity chain. Version C should not rewrite any of them.

The gaps cluster in four places, and each maps to a specific Version-C milestone:

1. **No LLM ever ran** (row 6, `OMITTED`) → C1–C3.
2. **Generator fairness was never established** (rows 14, 18, `NOT_ESTABLISHED` /
   `PARTIAL`) → C4 neutral GPAT-C and C6 matched 1024-sample banks. This is why B's H4 is
   not usable as an answer to the Version-C question.
3. **One protocol instead of three** (row 2, `PARTIAL`) → P1/P2 manifests, C8.
4. **Reliability failures were disclosed but not resolved** (row 34, `PARTIAL`) → the
   synthetic-vs-real probe becomes a blocking gate before P3.

Two rows are warnings against over-inheriting. Row 25: the multi-prototype manifold was
implemented exactly as specified and performed significantly *worse* — v1.1 turns it off
in the primary Track-R variant, and C7 must not quietly restore it. Row 28: the
calibration defect is the reason §16's "no fused score thresholded by a calibration fitted
on a different quantity" exists as a hard guard, and it must ship as a test, not a note.
