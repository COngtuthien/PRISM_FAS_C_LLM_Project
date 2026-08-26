# C9 detector-reliability decision dossier

**This is a scientific decision audit, not a scientific result.** No BA_sep
number, no residual-sensitivity number, and no other reliability metric was
computed, observed, loaded, or estimated while producing this document.
Nothing here was run scientifically. `target_access = 0` throughout. This
document recommends a protocol; it does not freeze one, and no code in this
repository was changed to make any option effective.

Audited at source HEAD: `721b917417a7fe319618e9871b4f13ced987c309`
(the last commit before this audit; see the git section of this session's
final report for the exact HEAD this dossier's own commit lands on).

Spec authority: `docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx`,
sha256 `ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5`
(matches `CLAUDE.md`'s recorded value; extracted and read section-by-section
for this audit — §3.1/3.1.1, §13, §16, §17, §18, §24, Appendix J/K/L).

---

## A. Exact current blocker

Confirmed from source, not assumed:

- `C9Adapter.required_inputs()` (`src/prism_fas/pipeline/adapters/c9.py:99-112`)
  declares `RequiredInput("detector_reliability_lock",
  detector_reliability.LOCK_PATH, ...)` where
  `detector_reliability.LOCK_PATH = "reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json"`.
- `C9Adapter.semantic_preconditions()` (`c9.py:114-137`) calls
  `detector_reliability.verify_lock(request.repo)` and reports the result as
  **blocking** whenever `verification["valid"]` is false.
- `detector_reliability.verify_lock()` (`src/prism_fas/evaluation/detector_reliability.py:210-249`)
  requires the lock file to exist, declare the frozen `schema_version` and
  `stage`, report `overall == "PASSED"`, resolve every one of the nine
  `REQUIRED_DETECTOR_RELIABILITY_TESTS` to `"PASSED"`, bind a non-empty
  `probe_protocol_identity` and `detector_checkpoint_identities`, and record
  `target_access == 0`. Today the lock file does not exist anywhere in this
  repository (development clone or GPU host), so `verify_lock` returns
  `valid=False` unconditionally and C9's precondition gate BLOCKS before
  `workflow()` — confirmed by `EngineeringAdapter.run()`
  (`src/prism_fas/pipeline/adapters/common.py:209-220`), which calls
  `full_precondition_gate` under a scientific context and returns its
  `BLOCKED` result without ever reaching `workflow()`.

Confirmed unresolved decision codes, read verbatim from
`detector_reliability.py:82-84`:

```
DETECTOR_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION
DETECTOR_BA_SEP_EVIDENCE_VECTOR_NEEDS_SCIENTIFIC_DECISION
DETECTOR_BA_SEP_PROBE_SEEDS_NEEDS_SCIENTIFIC_DECISION
```

`probe_protocol_status()` (`detector_reliability.py:135-156`) reports
`resolved = DETECTOR_BA_SEP_PROBE_PROTOCOL is not None`, and
`DETECTOR_BA_SEP_PROBE_PROTOCOL: dict[str, Any] | None = None` at module
scope. No code path in this repository sets it to anything else. None of
these three codes has been hard-coded to a pass by this audit.

**This is not a defect this task fixes.** It is a correctly-designed,
correctly-enforced fail-closed barrier, doing exactly what
`docs/PROJECT_STATE.md`'s `synthetic_vs_real_reliability_stage` decision
record (frozen 2026-08-23) says it should do.

---

## B. Spec-vs-implementation incompatibility (confirmed, not assumed)

Four independent statements in v1.5, none of them individually wrong, that do
not compose into an executable protocol as written:

1. §3.1.1 (verbatim): *"[C-H4] is evaluated after the three final C3 recipe
   banks and the common C6 synthetic gate are frozen, using source-only
   probe data."* — BA_sep runs **after** C6 closes.
2. §17's table row for the reliability barrier as a whole sits **before**
   "P3 target evaluation" in the milestone ordering, and §24's C6 milestone
   row (verbatim, `spec.txt` extraction) reads: *"3 synthetic BANK_LOCKs +
   reliability audit / Same final bank cardinality; **shortcut gates pass or
   STOP**."* — read literally, the reliability barrier's own gate belongs
   **at C6**.
3. The only canonical description of the synthetic-vs-real probe's feature
   space this project has ever recorded — inherited from Version-B, carried
   in `detector_reliability.py`'s own module docstring and
   `EVIDENCE_VECTOR_AUDIT` — is a linear probe over **detector evidence**:
   `p_global`, `s_region`, nine normalized regional distances.
4. C6 has no detector. C7 implements one; C8 trains it (confirmed:
   `src/prism_fas/detector/prism_detector.py`,
   `src/prism_fas/detector/trainer.py` — no detector model or checkpoint
   exists at or before C6 in this codebase).

(1)+(2) together are consistent (both put the barrier before target). But
(2)+(3)+(4) are not: a gate that must exist "at C6" cannot use "detector
evidence" that does not exist until C8. Building a NEW, separate image-level
bank probe purely to satisfy the literal C6 placement would invent a feature
extractor, classifier, split, training budget and seed policy v1.5 never
froze — exactly what Appendix K forbids ("Claude is expected to read
repository evidence to resolve inherited hashes/anchor values, not to invent
missing science").

**Already-resolved part** (not reopened by this audit):
`docs/PROJECT_STATE.md::synthetic_vs_real_reliability_stage`, frozen
2026-08-23, superseded the *placement* only: the barrier moves to
`SYNTHETIC_VS_REAL_RELIABILITY_STAGE = C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C`.
This did not choose a protocol, an evidence vector, or seeds, and was frozen
before any target access and before any C6 profile/bank result existed to
influence it.

**Not resolved, and the subject of this dossier**: the *executable protocol*
— what evidence vector, what probe seeds, what checkpoint(s), and every other
`PROBE_PROTOCOL_REQUIRED_FIELDS` entry.

---

## C. Evidence-vector audit

### C.1 What Track G actually emits (code-verified, not assumed)

`src/prism_fas/pipeline/adapters/c7.py:124-130`, `TRACK_G_FLAGS`:
`local_branch="off"`, `region="off"`, `manifold="off"`, `prototype_k=0`,
`prompt="off"`. §13.1/§13.4.1 of the spec are NORMATIVE and match exactly:
*"Track G MUST NOT instantiate ConvNeXt, region fusion, PromptHead or
manifold modules."*

`src/prism_fas/detector/contracts.py:130-181` (`ModelOutput.validate`):
`region_distances`, `region_valid`, `s_region` are all typed `| None` and
are `None` whenever no manifold is instantiated (`"without a manifold"`,
docstring line 133). `p_global` is the one field the contract asserts is
"always produced" (line 169-170).

**Conclusion**: Track G emits `global_logit_G` / `p_global` and nothing
else relevant here. It has never emitted, and by §13.1/§13.4.1's NORMATIVE
architecture contract can **never legitimately** emit, `s_region` or regional
distances under any authorized Track-G configuration.

### C.2 What primary Track R actually emits (code-verified)

`c7.py:131-134`, `TRACK_R_FLAGS = {**TRACK_G_FLAGS, "local_branch":
"convnext", "fusion": "glr_concat", "region": "on", "prompt":
"frozen_prompt"}` — **`manifold` is inherited unchanged from
`TRACK_G_FLAGS`, i.e. `"off"`.** This is the exact configuration C8's frozen
`C-R-DET` / `C-R-LLM` / `C-R-NOPROMPT` rows trained at (confirmed against
C7's own frozen `DETECTOR_CONFIG_LOCK.json` winner identities carried
forward into C8 — `docs/PROJECT_STATE.md::c7_scientific_closure_reconciliation.stage_table.c7.track_r`).

§13.2 (NORMATIVE): *"Track-R primary variant uses regions + PromptHead with
manifold OFF unless an explicit secondary row enables K=4."* §13.4.4's
target-time-input table states plainly: *"Manifold d_r / s_region — NO for
v1.3 primary Track R — Optional secondary regularizer/diagnostic only; no
post-hoc score fusion."*

**Conclusion**: with `manifold="off"`, primary Track R ALSO has
`region_distances = None` and `s_region = None` (same contract path as C.1).
It emits `local_logits`, `region_embeddings`, `fused_logit_R` / `p_R`, but not
regional *distances* — the specific quantity the historical vector needs.

### C.3 Is the historical vector obtainable from the existing 42 C8 rows, for RND, DET and LLM, without changing the scientific method?

**No, for two independent reasons, either one sufficient on its own:**

1. **No row in the 42-row matrix has manifold ON.** `TRACK_R_K4_FLAGS`
   (`c7.py:136-139`, `manifold="multi_prototype"`) is the only defined
   variant that would populate `region_distances` with 9 slots (matching
   `REGION_COUNT = 9`, `contracts.py:19`). `src/prism_fas/evaluation/source_matrix.py::plan_source_matrix`
   contains no row named for K=4, GCENTER, or any manifold-enabled Track-R
   variant — grepped directly, zero matches. `s_region`/`region_distances`
   are `None` for every one of the 42 trained checkpoints, in both tracks.
2. **Track R has no primary RND row at all.** §18.1's primary-rows table
   lists `C-R-DET` and `C-R-LLM` only; the C8 42-row matrix (given in this
   task's §0) has no `C-R-RND` entries. `EVIDENCE_VECTOR_AUDIT`
   (`detector_reliability.py:104-118`) already records this: *"Version-C
   Track-R primary rows are DET and LLM only... There is no preregistered
   Track-R RND row."* So even setting aside (1), `BA_sep_RND` computed from
   a Track-R representation is unobtainable from the existing matrix on arm
   coverage alone.

### C.4 Does C-R-RND (§18.2) solve this by itself? — verified, not assumed: **NO**

§18.2 lists `C-R-RND` as a recommended, not-yet-run diagnostic ("3 if compute
allows"). Even if run, it would use Track R's ONE frozen configuration per
the fairness invariant C7 already enforces (`c7.py`'s
`fairness_invariant`: *"ONE frozen configuration WITHIN a track, shared by
every primary generator arm of that track"*) — and that frozen configuration
has `manifold="off"`, per C.2. A hypothetical `C-R-RND` row trained under the
SAME frozen Track-R config would still emit `s_region = None`. It would close
the arm-coverage gap in C.3(2) alone, but not the manifold gap in C.3(1). To
close both, `C-R-RND` would have to run under a **different**,
manifold-enabled configuration — which is not what §18.2 names, and would
itself be a new, unpreregistered configuration choice.

### C.5 All six audited options

| # | Option | New training? | Changes 42-row matrix? | Alters C7 winners? | Preregistered? | Source-only? | Feature space identical across RND/DET/LLM? | Preserves BA_sep's intended meaning? | Compute cost | Scientific risk | User approval? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Common Track-G evidence** — `[global_logit_G, p_global]` (or `p_global` alone), computed for every Track-G checkpoint across RND/DET/LLM | **No** — uses the existing 15 Track-G C8 checkpoints | No | No | Would be, once approved | Yes | Yes (same 1-D quantity, same track, all 3 arms) | Weaker/reduced — 1-D instead of 11-D; still measures "does the detector's own decision evidence trivially separate synthetic from real" | Zero GPU (probe fit only, seconds on CPU) | Low — smaller feature space is a real reduction in probe power, must be disclosed, not hidden | **YES** (redefines the historical vector) |
| 2 | **Existing Track-R representation** — e.g. `[fused_logit_R, p_R]` and/or `local_logits`/`region_embeddings` (no `s_region`, manifold OFF) | No, for DET/LLM. **Yes** for RND (C.3(2)) | Adds `C-R-RND` if RND is required | No | Would be, once approved | Yes | Different space per track; and RND still needs new training | Weaker; also does not by itself fix arm coverage | Low-to-moderate (one new arm's rows if RND added) | Low-moderate — mixes two different feature spaces across tracks if used alongside Option 1 | **YES** |
| 3 | **C-R-RND diagnostic alone** (§18.2, at the SAME frozen manifold-OFF Track-R config) | **Yes** — 3 new source-only rows | No (additive diagnostic, not a change to the 42) | No | Named by §18.2, but not yet executed/preregistered as a BA_sep input | Yes | N/A — does not solve the manifold gap (C.4) | Does not, by itself, fix C.3(1) | Moderate (3 Track-R rows) | Low (uses an already-authorized config), but incomplete on its own | **YES** (as a BA_sep input, and to spend the compute) |
| 4 | **Manifold-enabled Track-R secondary** — K=4 (`TRACK_R_K4_FLAGS`, already typed in code) or GCENTER (`manifold="global_center"`, named at §18.2 as `C-R-GCENTER`, 1–3 seeds "source-only diagnostic") for RND, DET **and** LLM | **Yes** — none of RND/DET/LLM has a manifold-enabled row today; up to 9 new source-only rows (3 arms × 3 seeds) for whichever variant | No (additive) | No | K=4 is a typed secondary variant (§13.2) but never trained; GCENTER is named diagnostic (§18.2) but never trained. Neither is preregistered as a BA_sep input | Yes | Yes, within Track R, once trained for all 3 arms | Closest match to the historical 9-region vector (K=4 only; GCENTER gives 1 distance, not 9) | Highest of all options — up to 9 new full source-only training rows | Highest — §13.2 explicitly requires *"a new preregistered protocol version"* before a manifold-enabled `d_r`/`s_region` may be used for anything beyond training regularization | **YES**, explicitly required by §13.2's own text |
| 5 | **Shared already-existing frozen representation** — the frozen SigLIP2 image embedding `z_global` (input to BOTH tracks, always populated, `[B,768]`), or the primary calibrated score itself (`p_G`/`p_R`, §16.1) | **No** | No | No | Would be, once approved | Yes | `z_global`: yes, identical for every row in both tracks. `p_G`/`p_R`: no, different quantities per track | `z_global` measures the BACKBONE representation, not "the detector's own evidence" in Version-B's sense — a different question from the original probe's intent | Zero GPU | Low-moderate — changes what BA_sep measures more than Option 1 does (pre-detector features vs. post-detector decision evidence) | **YES** |
| 6 | **Declare C-H4 BLOCKED** | No | No | No | N/A | Yes | N/A | N/A — no BA_sep is produced | Zero | None (the fallback of last resort) | Only needed if the user rejects every recovery option |

No option was selected by looking at, estimating, or reasoning backward from
any BA_sep result — none exists, and none was computed to write this table.

---

## D. Probe-seed audit

- §3.1.1 (verbatim): *"three frozen source-only probe seeds"* — never names
  them.
- §18.3 (verbatim): *"Fixed seed family: 20260806–20260810 for 5-seed rows;
  first three seeds for 3-seed rows unless C0 records a deterministic
  replacement before any run."* — this is §18's **replication policy for
  hypothesis TRAINING rows** (C-H1 through C-H5's seeds/arm), which is a
  different population from the BA_sep probe.
- `src/prism_fas/evaluation/source_matrix.py:41-43`:
  `SEED_FAMILY = (20260806, 20260807, 20260808, 20260809, 20260810)`,
  `SEEDS_3 = SEED_FAMILY[:3]`. This is what the 42 C8 rows were actually
  trained at (confirmed: `docs/PROJECT_STATE.md` and this task's §0 give the
  matrix; every row's seed is drawn from this exact family). The module's
  own docstring (line 19-20) scopes it: *"The seed family is fixed at
  20260806-20260810... no 'best seed' exists to be chosen"* — about
  **training** rows.
- `detector_reliability.py::PROBE_SEED_AUDIT` (already in the frozen code,
  not written by this audit) states this exactly: the C8 seed-family policy
  *"is scoped to hypothesis TRAINING rows, and the probe is not a training
  row... whether the probe inherits that family is therefore not normative
  on the current audit."*

**Distinguished explicitly, as required:**

- **C8 TRAINING seeds** (already spent, already frozen, govern which of the
  42 checkpoints exist): `20260806`–`20260810` full family / first-3 subset,
  per row.
- **BA_sep PROBE seeds** (govern the probe's own train/validation split or
  fitting randomness, run AFTER a checkpoint already exists): **not
  authoritatively named by anything currently frozen.**

Using `20260806, 20260807, 20260808` for the probe would be a *reasonable*
choice — it is the exact family already frozen for 3-seed rows, so reusing it
introduces no new magic numbers — but it is **not inherited authority**: no
spec text or frozen record says the probe uses this family, or any family.

**Classification: `DETECTOR_BA_SEP_PROBE_SEEDS_NEEDS_SCIENTIFIC_DECISION`
remains USER_APPROVAL_REQUIRED.** No seed was chosen by, or after, observing
a BA value — none exists.

---

## E. All nine required reliability tests

Test *declarations* (population, measures, pass rule) are canonically owned
by `src/prism_fas/evaluation/reliability.py::DECLARED_TESTS` — the M10
shortcut/causal framework, already implemented and already used
structurally by C6's rehearsal (`c6.py::_reliability`,
`source="prism_fas.evaluation.reliability.declared_tests (canonical)"`).
`detector_reliability.py::REQUIRED_DETECTOR_RELIABILITY_TESTS` names which
nine of `reliability.py`'s ten declared tests must resolve `PASSED` at the
C8-closure barrier (the tenth, `benign_glasses_makeup_lowlight`, is
canonically blocked and excluded from the required set for a real,
documented reason — no legitimate benign population exists in this project).

| test_id | intended meaning | frozen source population (`reliability.py`) | detector/checkpoint population | perturbation | metric | threshold | aggregation | seed policy | existing impl? | fully frozen? | unresolved fields | new decision needed? | executable now? | target access |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `synthetic_vs_real_spoof_probe` | synthetic bank is not trivially fingerprintable | `source_train` real spoof vs. accepted M8 v3 synthetic spoof; no target | **UNRESOLVED — this dossier's central question** | none (a probe classifier, not a perturbation) | balanced accuracy of a synthetic-vs-real linear probe | `BA_sep <= 0.75` (§3.1.1, §17 — frozen) | mean over probe seeds (§3.1.1: "mean... over the three frozen... probe seeds") | **UNRESOLVED** (§D) | Declared (`reliability.py`); evidence-vector + checkpoint binding not implemented | No — evidence vector §C, checkpoint §F/§6, seeds §D all open | evidence_vector_definition, probe_seed_values, detector_checkpoint_identity, matched_source_split, class_balancing_rule, linear_probe_implementation, regularization, optimizer_or_solver, training_budget, train_validation_split | **YES** | No | NO |
| `residual_scale_zero` | score depends on the claimed synthetic artefact | `source_train` live, GPAT residual scaled to 0; no target | needs a Track-R (regional-distance-aware) checkpoint for "regional distance"; Track-G-only would report on score alone | scale the GPAT residual to zero | shift of spoof score and regional distance at zero residual | spec §17: movement SHOULD `>= 0.10` (frozen numeric ceiling exists) | not specified beyond the single frozen threshold | not seed-sensitive as declared (single deterministic intervention) | Declared, `score_shift()` helper implemented (`reliability.py:177-194`) | Threshold frozen; **detector/checkpoint binding not specified** | detector_checkpoint_identity, and (if regional distance is part of the metric) the same evidence-vector question as above | Partially — checkpoint binding | No (checkpoint unbound) | NO |
| `recipe_region_shift` | regional evidence localizes to the attacked region | `source_train` live under two recipes differing only in region; no target | requires Track R with a region-localized heat-map (region path is present with manifold OFF too — `region_embeddings` is populated) | swap the attacked region between two otherwise-identical recipes | displacement of the anomaly heat-map peak | **not numerically frozen** — §17: "moves with the attacked region" (qualitative) | not specified | not seed-sensitive as declared | Declared only | No — no numeric threshold anywhere in spec or code | pass_rule numeric definition, detector_checkpoint_identity | **YES** (no unique inherited threshold — see §5 note below) | No | NO |
| `artifact_map_swap` | local head is not decorative | accepted M8 v3 synthetic samples with artefact maps swapped; no target | Track R (needs a local/mask-aware supervision head) | swap the artefact map fed to local supervision | local-supervision performance under a mismatched map | **not numerically frozen** — §17: "performance drops" (qualitative) | not specified | not seed-sensitive | Declared only | No | pass_rule numeric definition, detector_checkpoint_identity | **YES** | No | NO |
| `cross_route_synthetic` | learned signal is not a single-generator fingerprint | train on one synthetic route, evaluate on the other; source only | either track; needs the checkpoint(s) trained per-route | train/eval route swap (Physics vs GPAT) | cross-route generalization of synthetic evidence | §17: "above preregistered source baseline" — **baseline not frozen anywhere found** | not specified | not seed-sensitive as declared | Declared only | No | baseline value, detector_checkpoint_identity, matched_source_split | **YES** | No | NO |
| `benign_jpeg_corruption` | detector does not read compression artefacts as attack evidence | `source_dev` LIVE only, benign JPEG re-encode; no spoof, no target | either track | benign JPEG re-encode | shift of spoof score on unchanged bona-fide content | §17: "bounded by preregistered source-only limits" — **limit not frozen anywhere found** | `score_shift()` reports mean/median/p95 shift (implemented) | not seed-sensitive | Declared + `score_shift()` implemented | No | numeric bound value, detector_checkpoint_identity | **YES** | No | NO |
| `benign_resize_corruption` | detector does not read resampling as attack evidence | `source_dev` LIVE only, benign resize; no spoof, no target | either track | benign resize | shift of spoof score | same as above — bound not frozen | `score_shift()` implemented | not seed-sensitive | Declared + implemented | No | numeric bound value, detector_checkpoint_identity | **YES** | No | NO |
| `benign_color_corruption` | detector does not read colour statistics as attack evidence | `source_dev` LIVE only, benign colour shift; no spoof, no target | either track | benign colour shift | shift of spoof score | same as above — bound not frozen | `score_shift()` implemented | not seed-sensitive | Declared + implemented | No | numeric bound value, detector_checkpoint_identity | **YES** | No | NO |
| `crop_padding_interpolation` | no crop-encoding shortcut | `source_dev` under different crop padding/interpolation; no target | either track | re-crop with different padding/interpolation | score sensitivity to a preprocessing-only change | §17: "below frozen threshold" — **threshold not found**; AND — | not specified | not seed-sensitive | Declared, but see next column | **STRUCTURALLY DATA-BLOCKED, already found in canonical code** | — | — | **Real gap, see note below** |

**Important note beyond what the task asked, found by faithful audit of
`reliability.py` (not invented for this dossier):** `crop_padding_interpolation`
is one of `detector_reliability.py`'s nine `REQUIRED_DETECTOR_RELIABILITY_TESTS`
— but the SAME canonical `reliability.py::DATA_BLOCKED` dict (line 136-145,
already in the frozen codebase, already exercised by `apply_execution`)
records it as structurally infeasible: *"the frozen package `prism_data_v1_m3b`
retains neither [a bbox column nor a path back to the source frame]...
re-deriving a crop from the stored crop would measure resampling (which
`benign_resize_corruption` already measures) rather than crop padding."*
This is a **package-level** limitation (what `prism_data_v1_m3b`'s
`manifests/samples.parquet` stores), independent of the C6→C8 staging move —
moving the barrier's stage does not restore the missing bbox/frame-path data.
As currently declared, this test can **never** resolve to `PASSED` — under
`barrier_state()` (`detector_reliability.py:159-207`), a `BLOCKED` result for
a *required* test lands in the `blocked` list and the barrier's `overall`
becomes `"BLOCKED"`, never `"PASSED"`, permanently. The only precedent this
codebase has for a genuinely-infeasible test
(`benign_glasses_makeup_lowlight`) was resolved by moving it OUT of the
required set into `CANONICALLY_BLOCKED_TESTS` — itself a scientific decision,
already made, already recorded. Whether `crop_padding_interpolation`
deserves the same treatment is **not decided by this audit** and is added to
§F as an additional open, result-unaffected item.

**On the un-frozen numeric thresholds** (§17's "bounded by preregistered
source-only limits" / "below frozen threshold" / "above preregistered source
baseline" language): searched `docs/PROJECT_STATE.md`, `DECISIONS.md`,
`configs/`, and Version-B lineage references reachable from this repository
for a uniquely inherited numeric value for `recipe_region_shift`,
`artifact_map_swap`, `cross_route_synthetic`, and the three benign-corruption
bounds. **None found.** Only `BA_SEP_CEILING = 0.75` and the residual
sensitivity `>= 0.10` figure are frozen numerics (both already correctly
carried in `detector_reliability.py`). Every other pass/fail numeric bound in
this table is classified `USER_APPROVAL_REQUIRED` rather than invented.

---

## F. Exact unresolved result-affecting fields

Every entry in `detector_reliability.py::PROBE_PROTOCOL_REQUIRED_FIELDS`,
classified per Appendix J's vocabulary — see §F.1 below for the full
per-field breakdown (this is a summary index).

**USER_APPROVAL_REQUIRED (11 of 20 protocol fields, plus 2 additional items
found by this audit):**

1. `evidence_vector_definition` (§C)
2. `probe_seed_values` (§D)
3. `detector_checkpoint_identity` (§6)
4. `matched_source_split` (definition, not just existence, is unresolved)
5. `class_balancing_rule`
6. `sample_unit`
7. `linear_probe_implementation`
8. `regularization`
9. `optimizer_or_solver`
10. `training_budget`
11. `train_validation_split`
12. **(additional)** numeric thresholds for `recipe_region_shift`,
    `artifact_map_swap`, `cross_route_synthetic`, and the three benign
    corruption tests (§E)
13. **(additional)** whether `crop_padding_interpolation` should be
    reclassified out of the required set the way
    `benign_glasses_makeup_lowlight` already was (§E)

**RECOVERED_FROM_CURRENT_FROZEN_CODE or FROZEN_BY_SPEC (already answerable,
no new decision) — see the full field table immediately below.**

None of these 13 open items may be resolved by observing a BA_sep, residual,
or any other reliability number — none exists.

### F.1 Every `PROBE_PROTOCOL_REQUIRED_FIELDS` entry, classified

Per Appendix J's own vocabulary (`FROZEN_CORE`/here `FROZEN_BY_SPEC` for a
directly-quoted spec value, `RECOVERED_FROM_VERSION_B`,
`RECOVERED_FROM_CURRENT_FROZEN_CODE`, `USER_APPROVAL_REQUIRED`). No field is
given a free-form default.

| field | classification | value / source |
|---|---|---|
| `real_spoof_population` | RECOVERED_FROM_CURRENT_FROZEN_CODE | `source_train` real spoof — `reliability.py:61` |
| `synthetic_population` | RECOVERED_FROM_CURRENT_FROZEN_CODE | accepted M8 v3 synthetic spoof — `reliability.py:61` |
| `source_domains` | RECOVERED_FROM_CURRENT_FROZEN_CODE | `[casia_fasd, msu_mfsd]` — the only two source domains this project has (§4.1 firewall; `source_matrix.py::PROTOCOLS`) |
| `matched_source_split` | USER_APPROVAL_REQUIRED | no unique inherited definition found; C-G4/C5 use `source_dev` for selection/calibration, but "matched" specifics for a probe are not named anywhere |
| `class_balancing_rule` | USER_APPROVAL_REQUIRED | not found in spec or code |
| `sample_unit` | USER_APPROVAL_REQUIRED | not found; M9's own decision score is frame-level by default, but this is not stated for the probe |
| `detector_checkpoint_identity` | USER_APPROVAL_REQUIRED | §6/§I — no checkpoint-selection rule exists anywhere; must not be chosen post-hoc |
| `evidence_vector_definition` | USER_APPROVAL_REQUIRED | §C — the central open question of this dossier |
| `preprocessing` | RECOVERED_FROM_CURRENT_FROZEN_CODE | identical to M9 training preprocessing (already frozen, `src/prism_fas/detector/trainer.py`) |
| `feature_normalization` | USER_APPROVAL_REQUIRED | no existing convention for a probe-specific feature vector |
| `linear_probe_implementation` | USER_APPROVAL_REQUIRED | sklearn is not a project dependency (checked `requirements/base.txt`, `requirements/cpu.txt` — absent) and no in-project logistic-regression code exists; §I |
| `regularization` | USER_APPROVAL_REQUIRED | not found |
| `optimizer_or_solver` | USER_APPROVAL_REQUIRED | depends on the implementation choice above |
| `training_budget` | USER_APPROVAL_REQUIRED | not found |
| `train_validation_split` | USER_APPROVAL_REQUIRED | not found |
| `probe_seed_values` | USER_APPROVAL_REQUIRED | §D — distinct from the C8 training-seed family, never named |
| `balanced_accuracy_implementation` | RECOVERED_FROM_CURRENT_FROZEN_CODE | `prism_fas.train.metrics.balanced_accuracy` — already implemented, `src/prism_fas/train/metrics.py:31-34`, no new dependency |
| `per_seed_aggregation` | RECOVERED_FROM_CURRENT_FROZEN_CODE | one BA value per probe seed, via the function above — implied by the metric's own definition |
| `mean_aggregation` | FROZEN_BY_SPEC | arithmetic mean over the three probe seeds — §3.1.1 verbatim: *"mean synthetic-vs-real probe balanced accuracy over the three frozen source-only probe seeds"* |
| `ba_ceiling` | FROZEN_BY_SPEC | `0.75` — §3.1.1 and §17 verbatim; already correctly carried as `BA_SEP_CEILING` in `detector_reliability.py:66` |

**11 of 20 fields are `USER_APPROVAL_REQUIRED`.** `evidence_vector_definition`,
`detector_checkpoint_identity` and `probe_seed_values` are the three the task
asked this audit to focus on (§3/§4/§6); the remaining eight
(`matched_source_split` through `train_validation_split`, excluding the
three already named) are genuinely open too and are not silently defaulted
by this dossier's recommendation in §H — they are proposed, not resolved,
in §I.

---

## G. All scientifically defensible resolution options

Given in full in §C.5 (evidence vector, six options). For seeds (§D): reuse
the frozen `20260806, 20260807, 20260808` family (recommended, not
inherited) or preregister a distinct probe-only family — both are equally
"scientifically defensible" in the sense that neither is forbidden; neither
is free.

---

## H. Recommended option, with rationale

**Recommended: Option 1 (§C.5) — a common Track-G evidence representation,
`[global_logit_G, p_global]` (or `p_global` alone), computed from the
existing 15 Track-G checkpoints (RND, DET, LLM), paired with the
already-frozen `20260806, 20260807, 20260808` seed subset reused explicitly
for the probe.**

Rationale:

- It is the **only** option in §C.5 that requires zero new training, touches
  none of the 42 existing C8 rows, and cannot possibly alter any C7 winner —
  it reads bytes that already exist.
- It is available identically for RND, DET and LLM **today**, because Track
  G is the only detector population with a primary row for all three arms
  (§C.3(2), §C.4) — Track R structurally cannot supply RND without new
  training, independent of the manifold question.
- `p_global` is architecturally guaranteed to exist under Track G by §13.1's
  NORMATIVE contract ("Track G MUST NOT instantiate... manifold modules"),
  so this choice cannot be undermined by a future Track-G configuration
  change the way a Track-R-manifold-dependent choice could be.
- It keeps the probe closest to Version-B's original *intent* — a linear
  probe over the trained detector's own decision evidence, not over a raw
  pre-detector backbone representation (Option 5's `z_global` alternative) —
  while being honest that it is a **reduced** evidence vector (1–2
  dimensions instead of 11) and therefore a **weaker** separability test
  than Version-B's. This tradeoff must be disclosed to and accepted by the
  user, not assumed.
- Options 3/4 (any C-R-RND or manifold-enabled Track-R work) remain
  available as a **later, separate, explicitly-approved** enrichment if the
  user wants a Track-R-specific or higher-dimensional BA_sep in addition —
  they are not foreclosed by adopting Option 1 now, and Option 1 does not
  require deciding against them.
- Option 6 (declare C-H4 BLOCKED) is not recommended, because Option 1
  demonstrably recovers a protocol without inventing new science, new
  training, or new architecture — the condition under which the task
  reserves Option 6.

**This recommendation is not effective.** `DETECTOR_BA_SEP_PROBE_PROTOCOL`
remains `None` in code; no field below is bound; the barrier remains
unresolved until the user approves.

---

## I. Exact protocol payload that WOULD be frozen if the user approves Option 1

Illustrative only — **not written to any config or lock by this audit**:

```yaml
DETECTOR_BA_SEP_PROBE_PROTOCOL:
  real_spoof_population: source_train real spoof (RECOVERED_FROM_CURRENT_FROZEN_CODE)
  synthetic_population: accepted M8 v3 synthetic spoof (RECOVERED_FROM_CURRENT_FROZEN_CODE)
  source_domains: [casia_fasd, msu_mfsd]           # matches C8's own matched-split domains
  matched_source_split: <TO BE DEFINED — proposed: source_dev, matched by domain and class,
                          identical to the split C-G4/C-G5 already use for selection/calibration>
  class_balancing_rule: <TO BE DEFINED — proposed: equal real-spoof vs synthetic-spoof count,
                          matching C6's own arm-balance convention>
  sample_unit: <TO BE DEFINED — proposed: frame, matching M9's frame-level decision score>
  detector_checkpoint_identity:
    interpretation: EVERY C8 Track-G checkpoint for the arm, per training seed
                    (20260806, 20260807, 20260808 subset of the arm's trained seeds);
                    BA_sep_arm = mean probe balanced accuracy over the 3 PROBE seeds,
                    itself averaged over however many C8 TRAINING-seed checkpoints
                    exist for that arm — no single "representative" checkpoint chosen
    binds: [checkpoint_sha256 of every C-G-{RND,DET,LLM} row's best.pt]
  evidence_vector_definition:
    fields: [global_logit_G, p_global]
    source: Track-G ModelOutput, always populated per contracts.py:130-181
    note: reduced relative to Version-B's [p_global, s_region, 9 regional distances];
          Track G structurally cannot produce the regional terms (§13.1)
  preprocessing: identical to M9 training preprocessing (RECOVERED_FROM_CURRENT_FROZEN_CODE)
  feature_normalization: <TO BE DEFINED — proposed: none needed for a 1-2D vector, or
                          z-score per feature fit on the probe's own training split only>
  linear_probe_implementation: <TO BE DEFINED — sklearn is NOT a project dependency
                                (checked requirements/base.txt, requirements/cpu.txt: absent) and
                                no existing linear-probe/logistic-regression code exists in this
                                repository. Two honest options, neither free: (a) add scikit-learn
                                as a new dependency and use LogisticRegression, or (b) implement a
                                small numpy/torch logistic-regression fit in-project, consistent
                                with prism_fas.train.metrics' existing numpy-only style. This
                                choice is itself part of what needs approval.>
  regularization: <TO BE DEFINED — proposed: L2, default strength, given the reduced
                   feature dimensionality (1-2D) makes overfitting unlikely>
  optimizer_or_solver: <TO BE DEFINED — depends on the linear_probe_implementation choice above
                        (sklearn lbfgs, or a small closed-form/gradient-descent fit in-project)>
  training_budget: <TO BE DEFINED — proposed: full-batch closed-form/lbfgs fit, no epochs>
  train_validation_split: <TO BE DEFINED — proposed: same matched_source_split, k-fold or
                           held-out fraction fixed before the first probe seed runs>
  probe_seed_values: [20260806, 20260807, 20260808]   # reused C8 family; NOT inherited authority
  balanced_accuracy_implementation: prism_fas.train.metrics.balanced_accuracy
                                    (RECOVERED_FROM_CURRENT_FROZEN_CODE — already implemented,
                                    src/prism_fas/train/metrics.py:31-34, numpy-only, no new
                                    dependency)
  per_seed_aggregation: one BA value per probe seed, via balanced_accuracy() above
                        (RECOVERED_FROM_CURRENT_FROZEN_CODE — the metric's own definition)
  mean_aggregation: arithmetic mean across the 3 probe seeds (FROZEN_BY_SPEC, §3.1.1)
  ba_ceiling: 0.75   # FROZEN_BY_SPEC, §3.1.1 and §17
```

Every `<TO BE DEFINED — proposed: ...>` field is a genuine open decision;
the "proposed" text is this audit's recommendation, not a claim of existing
authority.

---

## J. Exact additional source-only compute required

**Under the recommended Option 1: zero additional GPU training.** The probe
fits a 1–2 dimensional linear classifier over already-computed
`global_logit_G`/`p_global` values from the 15 existing Track-G checkpoints
— a CPU-scale operation (seconds), not a training run. No new C8 row, no new
C7 search, no retraining.

If the user instead approves any Track-R-inclusive option (§C.5 options 2-4),
additional compute is:

- Option 2/3 (`C-R-RND` alone): 3 new source-only Track-R rows at the
  existing frozen (manifold-OFF) Track-R config — comparable per-row cost to
  the 9 existing Track-R rows already in C8.
- Option 4 (manifold-enabled Track-R for all three arms): up to 9 new
  source-only rows (3 arms × 3 seeds) at a NEW Track-R configuration
  (`TRACK_R_K4_FLAGS` or a GCENTER variant), plus the one-time engineering
  cost of wiring a new frozen search/finalize path for that configuration if
  one does not already exist end-to-end.

---

## K. Exact implementation files that WOULD change

If Option 1 is approved, at minimum:

- `src/prism_fas/evaluation/detector_reliability.py` — bind
  `DETECTOR_BA_SEP_PROBE_PROTOCOL` to the frozen payload (§I), replacing
  `None`; this is the ONLY line in this module the task's confirmed
  unresolved codes gate on.
- A new module or function (not yet named) implementing the linear probe fit
  and BA_sep computation itself, reading `global_logit_G`/`p_global` from
  the existing C8 checkpoint/manifest evidence via the same
  `source_evidence`/`RowEvidence` machinery C9 already uses to read C8 rows
  read-only (`src/prism_fas/evaluation/source_evidence.py`,
  `src/prism_fas/evaluation/source_lock.py::RowEvidence`) — reused, not
  reimplemented.
- A writer for `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`, using
  `detector_reliability.lock_payload()` (already implemented,
  `detector_reliability.py:252-265`) — no new lock-writing logic needed,
  only the caller that supplies real `results`, `probe_protocol_identity`
  and `detector_checkpoint_identities`.
- Possibly a new C8-adjacent adapter substage or a standalone script,
  depending on how the user wants this sequenced relative to C8's own
  ACCEPTANCE — **not decided here.**

**No file in this list was changed by this audit.**

---

## L. Exact tests required before execution

- A unit test proving `probe_protocol_status()["resolved"]` becomes `True`
  only once every `PROBE_PROTOCOL_REQUIRED_FIELDS` entry is bound (already
  partially covered by existing `detector_reliability` behavior — the module
  refuses `None`; needs an explicit regression once a real payload exists).
- A unit test proving the probe's evidence-vector extraction never reads
  `s_region`/`region_distances` for Track G (should always be `None` and
  must not be silently coerced to zero — `contracts.py`'s own contract
  already forbids fabricated zeros for absent terms).
- A unit test proving the probe never opens `sources.target_roots` or any
  SiW-Mv2 path — mirroring the existing target-firewall regressions in
  `tests/pipeline/test_scientific_fixture_leakage.py` and the C9/C10 target
  isolation tests.
- A regression proving `verify_lock()` still requires `overall == "PASSED"`
  and every required test `== "PASSED"` — i.e. that implementing the probe
  does not loosen `verify_lock`'s existing strictness.
- A regression proving the lock's `probe_protocol_identity` changes if and
  only if a result-affecting field of the bound protocol changes (mirrors
  the existing pattern for `c7_search_decision`/`lr_decision` identities).
- End-to-end: C9's `semantic_preconditions` correctly transitions from
  BLOCKED to non-blocking once a real, valid lock exists — exercised the
  same way `test_c8_precondition_root_drift.py` proved C8's gate reacts to
  its own canonical verifier.

None of these tests exist yet and none was written by this audit — writing
them is future work gated on the user's approval of a protocol.

---

## M. No reliability metric was observed

**NO RELIABILITY METRIC WAS OBSERVED WHILE MAKING THIS RECOMMENDATION.** No
BA_sep value, no residual-sensitivity value, no corruption-shift value, and
no probe balanced-accuracy value was computed, loaded, estimated, or
inspected at any point during this audit. The recommendation in §H follows
only from: (a) which tensors the frozen architecture can structurally
produce (§C.1, §C.2 — code-verified contract facts, not measurements), and
(b) which C8 rows already exist (§C.3 — a row-count/arm-coverage fact, not a
performance measurement).

## N. Target access

`target_access = 0`. No SiW-Mv2 path, manifest or label was opened, resolved,
or referenced by anything this audit read or wrote. Every file inspected was
under `src/`, `docs/`, `configs/`, `reports/readiness/`, or `docs/PROJECT_STATE.md`.

---

## Appendix: files audited

```
src/prism_fas/pipeline/adapters/c9.py
src/prism_fas/evaluation/detector_reliability.py
src/prism_fas/evaluation/reliability.py
src/prism_fas/evaluation/source_matrix.py
src/prism_fas/evaluation/source_lock.py            (RowEvidence shape only)
src/prism_fas/evaluation/source_evidence.py         (referenced, not modified)
src/prism_fas/detector/contracts.py                 (ModelOutput contract)
src/prism_fas/detector/variant.py                   (manifold flag domain)
src/prism_fas/detector/prism_detector.py            (manifold docstring only)
src/prism_fas/pipeline/adapters/c7.py               (TRACK_G_FLAGS/TRACK_R_FLAGS/TRACK_R_K4_FLAGS)
src/prism_fas/pipeline/adapters/c6.py               (reliability.py usage at C6 rehearsal)
src/prism_fas/pipeline/adapters/common.py           (EngineeringAdapter.run/full_precondition_gate)
configs/experiments/m10_matrix.yaml                 (manifold: global_center precedent)
docs/PROJECT_STATE.md                               (synthetic_vs_real_reliability_stage record)
docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx  (§3.1/3.1.1, §13, §16, §17, §18, §24, App. J/K/L)
```
