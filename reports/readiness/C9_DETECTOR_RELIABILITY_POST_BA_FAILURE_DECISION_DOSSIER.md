# C9 detector reliability — post-BA_sep-failure decision dossier

This is a decision dossier, not a redesign. **Nothing in this document
implements a redesign.** It explains the current scientific situation
precisely, audits what remains available under the CURRENT frozen protocol,
and lays out three explicit, mutually-exclusive options for what happens
next — without choosing or implementing any of them beyond a bounded
recommendation for which to consider first.

---

## A. The current scientific situation

`synthetic_vs_real_spoof_probe` — the first of nine
`REQUIRED_DETECTOR_RELIABILITY_TESTS` — has been scientifically executed,
exactly once, under the frozen `C9_DETECTOR_BA_SEP_OPTION1_V2` protocol, and
has **FAILED**:

```
BA_sep_RND = 0.7843079833902619   (ceiling 0.75)
BA_sep_DET = 0.8514170182841069   (ceiling 0.75)
BA_sep_LLM = 0.7902658339472685   (ceiling 0.75)
```

Full numeric detail, identities, and the no-rerun/registration mechanics are
in `reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.md`;
this dossier does not repeat them.

## B. The consequence under the currently frozen Version-C protocol

`detector_reliability.barrier_state`'s rule — unchanged, unweakened, not a
new decision — is: **if any required test is FAILED, `overall = FAILED`,
full stop.** This is not a majority vote and not a weighted score. Under
this rule:

```
DETECTOR_RELIABILITY_LOCK_C cannot PASS
C9 SOURCE_MATRIX_LOCK_C cannot close
C10 cannot proceed
C11 cannot proceed
target prediction remains forbidden
```

**Even if all remaining eight required tests later PASS, this does not
change.** One required FAILED test is sufficient, and sticky: resolving
eight open questions cannot un-fail the ninth. This is stated explicitly
because it is the single most important fact this dossier records — no
amount of additional source-only work under the CURRENT protocol version
can reopen C9. Reopening it requires either accepting the negative result
as terminal (Option A) or a new, separately versioned protocol (Option C).
Diagnostic work (Option B) illuminates WHY the failure occurred; it cannot,
by construction, reverse it.

## C. The remaining eight required tests — current status audit

Audited against `src/prism_fas/evaluation/reliability.py` (the M10
declarative framework) and `src/prism_fas/pipeline/adapters/c6.py`
(`DETECTOR_LEVEL_RELIABILITY_TESTS`), which already correctly stages all
eight as requiring a trained detector (post-C7/C8, same stage as the BA_sep
probe). None was run. None is classified `PASSED` or `FAILED` anywhere in
this repository.

| test | classification | why |
|---|---|---|
| `residual_scale_zero` | `NEEDS_SCIENTIFIC_DECISION` | `reliability.py` declares only a prose pass rule ("score approaches live and the regional distance falls") — no frozen numeric threshold, no frozen scaling protocol, no implementation exists anywhere outside the declaration. It also plausibly needs REGIONAL detector evidence (`s_region`/regional distance), the same evidence class the BA_sep audit already found absent from every primary Version-C row (Track G is global-only by NORMATIVE contract; primary Track R trains with manifold OFF) — this needs its own audit before a protocol can even be proposed, not assumed away. |
| `recipe_region_shift` | `NEEDS_SCIENTIFIC_DECISION` | Same gap: a prose pass rule ("the heat-map moves with the attacked region"), no frozen displacement metric or threshold, no implementation, and it needs a REGIONAL anomaly heat-map — the same likely-absent evidence class. |
| `artifact_map_swap` | `NEEDS_SCIENTIFIC_DECISION` | Needs "local supervision performance" — a local/regional head output. No frozen protocol, no implementation, and the same regional-evidence question applies. |
| `cross_route_synthetic` | `NEEDS_SCIENTIFIC_DECISION` | Pass rule ("performance is retained across routes") names no metric, no threshold, no frozen route pairing, no implementation. |
| `benign_jpeg_corruption` | `NEEDS_SCIENTIFIC_DECISION` | The MEASUREMENT primitive exists and is reusable (`reliability.score_shift` — paired mean/median/p95 shift, already implemented, generic). What is NOT frozen: the JPEG quality/re-encode parameter, the population size/seed, and — critically — what counts as "no systematic large increase" numerically. Closer to executable than the four above, but still not `EXECUTABLE_WITH_FROZEN_PROTOCOL`. |
| `benign_resize_corruption` | `NEEDS_SCIENTIFIC_DECISION` | Same as above: `score_shift` reusable, resize factor/threshold not frozen. |
| `benign_color_corruption` | `NEEDS_SCIENTIFIC_DECISION` | Same as above: `score_shift` reusable, color-shift magnitude/threshold not frozen. |
| `crop_padding_interpolation` | `STRUCTURALLY_DATA_BLOCKED` | See §D. |

No threshold was invented for any of these while writing this audit — every
"not frozen" claim above is a claim about what does NOT exist in this
repository today, checked by reading `reliability.py` and grepping the
codebase, not a judgment about what the right threshold WOULD be.

## D. `crop_padding_interpolation` — re-audited, still structurally data-blocked

Preserved from the prior audit
(`src/prism_fas/evaluation/reliability.py::DATA_BLOCKED["crop_padding_interpolation"]`,
re-read as part of this dossier, unchanged): the frozen `prism_data_v1_m3b`
package's `manifests/samples.parquet` stores the 224×224 CROP and its hash —
no bounding-box column, no path back to the original source frame.
Reproducing the intended crop-padding/interpolation perturbation needs the
original frame and the detected bounding box, neither of which the frozen
package retains. Re-opening the raw video tree is outside what M10 may
touch. Re-deriving a "crop padding" perturbation FROM the already-cropped
224×224 image would measure resampling — which `benign_resize_corruption`
already measures — not crop padding; that would answer a different question
under the same name.

**Classification confirmed: `STRUCTURALLY_DATA_BLOCKED`.** Not fabricated
with synthetic original frames, not inferred bounding boxes, not
substituted with a different perturbation, not marked PASS.

## E. Three explicit, mutually exclusive post-failure options

### Option A — close current Version C as negative

Preserve the `BA_sep` FAILED result as terminal. Do not execute the
remaining eight tests unless purely for reporting/paper completeness (their
outcome cannot reopen C9 either way, per §B). Current Version C does not
reach target evaluation. Scientifically clean — the cleanest possible
closure — but it terminates this protocol's path to target with no further
source-only investment.

### Option B — complete source-only diagnostics (bounded, non-reopening)

Preserve the terminal `BA_sep` FAILED result — unchanged, un-reopened.
Freeze EXECUTABLE protocols (thresholds, populations, seeds — the same kind
of freeze `Option-1 V2` itself required) for the remaining source-only
tests, and run them purely for DIAGNOSTIC/MECHANISTIC understanding: which
of residual strength, recipe locality, artifact mapping, route-specific
shortcut, JPEG sensitivity, resize sensitivity, or color sensitivity is more
consistent with WHY the shortcut separability is this high. Their outcomes
cannot reopen C9 under current Version C (§B) — this is diagnostic evidence
for a paper and for redesign justification, not a second chance at the same
gate. Zero target access throughout.

### Option C — new, explicitly versioned pre-target redesign protocol

Preserve the current Version-C negative result PERMANENTLY, as immutable
historical evidence — never mutated, never superseded in place. Create a
NEW, explicitly versioned scientific protocol (a genuinely new decision ID,
never a V3 of Option-1, never a silent edit of
`c9_detector_ba_sep_option1_v2.yaml`). No retroactive change to the current
protocol's threshold, seeds, or population. Any redesign work — a new
evidence representation, a new generator, a new detector configuration —
happens SOURCE-ONLY, before any target access, and any regenerated bank,
detector, or checkpoint receives entirely NEW identities (never reusing or
overwriting the current C6/C8 identities). The current C8 result and this
BA_sep result remain immutable historical evidence regardless of what a new
protocol later finds. Target stays sealed until the NEW protocol's own
reliability gates pass on their own terms.

**Option C is explicitly NOT implemented by this task.** This dossier
compares it; it does not build it.

## F. Recommendation

**Run Option B first — bounded, source-only — before deciding whether
Option C is scientifically justified.**

Reason: the BA_sep failure already establishes that synthetic-vs-real
shortcut separability is too high under the frozen Option-1 V2 evidence
representation. What it does NOT establish is WHY. Bounded source-only
diagnostics under the remaining tests can distinguish whether the
separability is more consistent with residual strength, recipe locality,
artifact-map mismatch, a route-specific shortcut, or corruption sensitivity
— evidence that materially changes what a Option-C redesign should even
target, versus proceeding to redesign on pure speculation. This
recommendation is explicitly conditional and bounded:

- Option B does **NOT** rescue current C9. Re-read §B: no diagnostic result
  reopens the barrier under the current protocol version, regardless of
  outcome.
- Option B's own protocols must themselves be frozen BEFORE execution,
  exactly as Option-1 V2 was — this dossier does not authorize skipping that
  discipline for the remaining eight tests.
- If Option B's diagnostics suggest the shortcut is structural to the
  current synthetic generation approach (rather than an artifact of the
  probe's evidence representation), that is exactly the evidence that would
  justify moving to Option C rather than repeating Option A/B indefinitely.

## G. What this dossier does NOT do

- Does not silently start a redesign, of any kind.
- Does not implement Option B's protocols.
- Does not implement Option C.
- Does not invent a threshold for any of the eight unresolved tests.
- Does not touch `configs/evaluation/c9_detector_ba_sep_option1_v2.yaml`.
- Does not create a V3 protocol.
- Does not access target data.
- Does not change `DETECTOR_RELIABILITY_LOCK_C`'s meaning or weaken
  `verify_lock`.

`target_access = 0` throughout this dossier and everything it references.
