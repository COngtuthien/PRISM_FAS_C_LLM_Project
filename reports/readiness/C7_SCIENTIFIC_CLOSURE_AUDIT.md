# C7 scientific closure audit — `inherited_anchor_report` is a pre-decision
# diagnostic, not the final scientific state

**This is a code-path audit and a reporting-semantics fix, layered on top of
scientific evidence that was produced elsewhere.** The completed GPU C7 run's
scientific facts (Track G, Track R, winner identities, checkpoint hashes,
`target_access = 0`) are asserted as frozen, immutable evidence — this document
does not re-derive them, does not re-run them, and this laptop has no copy of
`reports/full/c7/DETECTOR_CONFIG_LOCK.json` to re-hash (the same situation
`docs/PROJECT_STATE.md` already records for `reports/full/c6`: it lives on the
GPU host). What this document DOES independently establish, from the source
code and from an offline CPU rehearsal of the real C7 production lifecycle
(`tests/pipeline/test_c7_scientific_path.py`), is the exact semantics of the
field this audit was asked to resolve, and that nothing scientific moved while
fixing its presentation.

Audit source HEAD: `17a401651d96226c7e04933d98b1534f19e26605`
Fix source HEAD: see §7 (git) below.

---

## 1. Audit finding

### 1.1 Where `inherited_anchor_report` comes from

`anchor_resolution_report()` (`src/prism_fas/search/plan.py`) is a **pre-decision
structural diagnostic**, shared verbatim by C4's GPAT envelope and C7's detector
envelope. It is built from `resolve_anchors()`, which looks up each coordinate's
candidate scalar paths (e.g. `optimizer.learning_rate`,
`optimizer.backbone_lr`, `optimizer.head_lr`) directly in the **raw inherited
Version-B configuration**, before any approved decision is applied:

```
detector_search_plan()
  resolutions = resolve_anchors(anchors, paths)        # RAW, pre-decision
  coordinates = [coordinate_from_resolution(...) ...]  # still pre-decision
  coordinates, base_config = _apply_lr_decision(coordinates, base_config, lr_decision)
                                                         # <- decision applied HERE,
                                                         #    to `coordinates`, NOT
                                                         #    to `resolutions`
  plan = SearchPlan(coordinates=coordinates, ...)
```

`_apply_lr_decision` (`plan.py`) replaces the ambiguous per-scalar
`learning_rate` coordinate **in place** with the single
`learning_rate_multiplier` coordinate the approved decision authorizes. It
mutates `coordinates` — what becomes `plan.coordinates`, what the search engine
actually walks, and what `SearchPlan.identity_material()` hashes. It does
**not** touch `resolutions`, the dict `anchor_resolution_report()` was already
called on at the caller site (`c7.py::_scientific_plan`, before
`_apply_lr_decision` runs inside `detector_search_plan`).

The consequence: **both tracks' inherited M9 configuration declares two
candidate LR scalars — `optimizer.backbone_lr` and `optimizer.head_lr` — so
`resolve_anchors` always finds more than one hit for `learning_rate` and always
classifies it `AMBIGUOUS`.** `anchor_resolution_report()` therefore reports
`ambiguous: ["learning_rate"]` and `executable_under_full: false` for **every**
real C7 plan build, Track G and Track R alike, regardless of whether an
approved LR decision has since resolved that exact ambiguity. This is not a
bug in the raw lookup — §15.2.2 genuinely says a non-unique inherited scalar is
`USER_APPROVAL_REQUIRED` — but the lookup answers "is this executable with NO
decision", not "is this executable with the decision that was actually
approved and applied".

### 1.2 Where it lands, unlabeled, in the FINAL lock

`c7.py::_finalize_track` builds the per-track `resolved` dict that becomes
`DETECTOR_CONFIG_LOCK.json["tracks"][track]`. Before this fix, that dict wrote:

```python
"lr_interpretation": item["lr"].interpretation,   # the FINAL, resolved state
"lr_anchor_vector": dict(item["lr"].anchor_vector),
"inherited_anchor_report": item["anchor_resolution"],  # the RAW, pre-decision state
```

Both the resolved and pre-decision views were present, but **nothing labeled
which was which.** A reader (or an automated verifier that is not
`verify_detector_config_lock` — see §1.3) encountering
`inherited_anchor_report.executable_under_full = false` and its
`blocking_reason` — literally "the full profile cannot execute this envelope
until the user binds each one to a named scalar" — sitting inside a lock
section that also names a `winner_config`, a `winner_checkpoint_sha256` and 15
or 24 retained trials, has no structural signal telling them the blocking
claim predates and is superseded by the LR decision two keys above it.

### 1.3 Whether anything scientific reads it

Grepped the full `src/prism_fas` tree for every consumer of
`inherited_anchor_report`, `executable_under_full`, `ambiguous` and
`blocking_reason`:

- **Producers only:** `search/plan.py` (defines the keys) and
  `pipeline/adapters/c7.py:2088` (the one write site into the final lock).
  `pipeline/checks.py:551` uses `report["executable_under_full"]` for the C4
  rehearsal check `c4_plan_executable_under_full` — computed fresh from the
  live `report`, not read back out of a stored artifact.
- **No reader.** `verify_detector_config_lock` (`c7.py`, shared module-level
  with C8) does not reference any of these four names — its per-track checks
  gate on `search_plan_identity`, `coordinate_order`, `selection_tuple`,
  `tie_break`, `variant_identity`, `retained_trials`, `winner_config*`,
  `winner_checkpoint*`, `decision_logit_name`, `decision_score_name` and
  `decision_graph_hash`.
- **C8 reads only the resolved state.** `c8.py::track_configuration` returns
  `sub["winner_config"]`; `c8.py::_track_parents` reads
  `winner_config_sha256`, `search_plan_identity`, `decision_graph_hash` (plus
  lock-level `search_decision_identity`, `training_arm`). Neither touches
  `inherited_anchor_report` or any of its inner keys.
- **The C7-internal EXECUTABILITY gate is separate code.** Whether a track's
  plan is blocked (`c7_track_{g,r}_plan_executable`) is computed from
  `plan.coordinates` — the POST-`_apply_lr_decision` coordinates — via
  `blocked = [c.name for c in plan.coordinates if not c.applicable and
  "AMBIGUOUS" in str(c.skip_reason)]`. Since the LR coordinate was already
  replaced by `learning_rate_multiplier` (applicable) before this check runs,
  `blocked` is empty whenever the decision resolved the ambiguity — the gate
  never reads `report["executable_under_full"]`.

**Conclusion: this is Case A (an intentional, legitimate pre-decision
diagnostic) presented without an explicit phase label, which is close enough
to Case B (a final-state reporting bug) that a human or a downstream tool
reading the frozen lock in isolation could misread it as a live blocker.** No
scientific decision — not C7's own gating, not `verify_detector_config_lock`,
not C8's row configuration — was ever computed from it. The defect is
purely one of presentation inside evidence that is otherwise correct.

### 1.4 Exact files/functions

| File | Function / site |
|---|---|
| `src/prism_fas/search/plan.py` | `AnchorResolution`, `resolve_anchors`, `_apply_lr_decision`, `gpat_search_plan`, `detector_search_plan`, `anchor_resolution_report` |
| `src/prism_fas/pipeline/adapters/c7.py` | `_scientific_plan` (builds `report` per track, pre-decision), `_finalize_track` (embeds it, unlabeled, into the final lock), `_scientific_finalize` (writes `DETECTOR_CONFIG_LOCK.json`), `verify_detector_config_lock` (never reads it) |
| `src/prism_fas/pipeline/checks.py:551` | `check_c4_search_plan` — the one other consumer, C4's own rehearsal-only diagnostic |
| `src/prism_fas/pipeline/adapters/c8.py` | `track_configuration`, `_track_parents` — confirmed to read only `winner_config*` / identity fields |

### 1.5 C8 dependency impact

None. §1.3 establishes C8 never reads `inherited_anchor_report` or its inner
keys. C8's precondition gate re-runs the exact same
`verify_detector_config_lock` C7 used to write the lock, so a lock produced by
the pre-fix code (the actual GPU-host lock, which this laptop cannot see)
verifies identically before and after this fix — see §3 for the regression
proving that.

---

## 2. Scientific impact

| Question | Answer |
|---|---|
| C7 trials changed? | **NO** |
| Winners changed? | **NO** |
| Checkpoints changed? | **NO** |
| Metrics changed? | **NO** |
| Search identities changed? | **NO** |
| Target access | **0** (unchanged, unauthorized code path throughout) |
| Retraining required? | **NO** |

`anchor_resolution_report()`'s output and `_finalize_track`'s new
`lr_decision_resolution` / `lr_decision_identity` fields never enter
`SearchPlan.identity_material()` — that method hashes only
`schema_version`, `plan_id`, `milestone`, `coordinate_order`,
`coordinates` (i.e. `Coordinate.as_dict()` for each POST-decision
coordinate), `selection_tuple`, `tie_break`, `one_pass`,
`revisit_permitted`, `base_config` and `lock_deadline`. None of those fields
reference the raw `resolutions` dict or the report built from it.
`tests/pipeline/test_c7_scientific_path.py::test_reporting_metadata_never_enters_the_search_plan_identity`
asserts this directly by serializing `plan.identity_material()` and checking
none of the new/changed diagnostic keys appear in it. Winner identity, trial-set
digest and `decision_graph_hash` are independently derived from the trained
checkpoint and the trial evidence, never from this report.

---

## 3. Fix

**Files changed:**

- `src/prism_fas/search/plan.py` — `anchor_resolution_report()` gains two
  additive keys: `diagnostic_scope: "PRE_DECISION_STRUCTURAL"` and a
  `scope_note` explaining that `ambiguous` / `executable_under_full` /
  `blocking_reason` describe the raw inherited configuration, not whether a
  bounded scientific search subsequently executed. No existing key's value or
  meaning changed; `tests/pipeline/test_search_engine.py`'s existing assertions
  on `resolved` / `absent` / `ambiguous` / `executable_under_full` for the
  genuinely-unresolved C4 case are untouched and still pass.
- `src/prism_fas/pipeline/adapters/c7.py::_finalize_track` — the per-track lock
  section keeps `inherited_anchor_report` verbatim (now carrying the new scope
  labeling above), and adds two new sibling keys carrying the FINAL, resolved
  state explicitly: `lr_decision_identity` (the approved
  `LRDecisionRecord.identity` this track trained under) and
  `lr_decision_resolution` (the full `LRAnchorDecision.as_dict()` — interpretation,
  whether the coordinate was searched, the multiplier set, the per-multiplier
  learning rates, the preserved ratio, whether the anchor trial reproduces
  Version B). A code comment at the write site cross-references the two so a
  future reader cannot land on `inherited_anchor_report` without also seeing
  the note pointing at the resolved fields.

**Why it cannot affect scientific selection:** every field touched is
diagnostic metadata written into `C7_SCIENTIFIC_SEARCH_PLAN.json` (plan stage)
and `DETECTOR_CONFIG_LOCK.json` (finalize stage) *alongside* the fields that
already carried the winner, the checkpoint and the trial evidence — nothing
was removed, nothing existing was renamed, and (§2) none of it is part of any
identity, any hash, or any check that gates C7 or C8 execution. This is
strictly additive, backward-compatible reporting: a consumer reading only the
pre-fix keys (`lr_interpretation`, `lr_anchor_vector`, `winner_config*`,
`search_plan_identity`, `decision_graph_hash`, …) sees byte-identical values.

**Why the already-frozen GPU lock was not regenerated:** per the safe
resolution policy, rewriting `reports/full/c7/DETECTOR_CONFIG_LOCK.json` would
require re-running `FINALIZE_DETECTOR_CONFIG` against real trial evidence on
the GPU host — a scientific execution step, not a finalization-only or
verify-only path (`_scientific_finalize` reads the trial summaries and
checkpoints `_run_scientific_trial` already wrote, but the stage's own
`workflow()` dispatch reaches it only via the full `_scientific_workflow`
sequence, and this laptop cannot construct or check the real trial evidence
those checks require). It is also unnecessary: §1.3 proves the pre-fix lock
already verifies cleanly under `verify_detector_config_lock`, and this fix adds
new fields nothing downstream requires. **The GPU-host lock is left exactly as
the completed run wrote it.** This document is the separate closure/consistency
evidence record the policy calls for when the frozen artifact itself is not
touched.

---

## 4. Tests

**Focused (offline, no GPU, no live provider, no target access):**

```
/…/testenv/bin/python3 -m pytest \
  tests/pipeline/test_c7_scientific_path.py \
  tests/pipeline/test_lr_track_g_coordinate.py \
  tests/pipeline/test_search_engine.py \
  tests/pipeline/test_c8_scientific_path.py \
  -q -p no:cacheprovider
```
→ **105 passed**, 0 failed.

New assertions added to `tests/pipeline/test_c7_scientific_path.py`, all
driving the REAL `_scientific_plan → _scientific_search → _scientific_finalize
→ verify_detector_config_lock` lifecycle end to end (the same `_StubTrainer`
harness the file already uses — stubs only "put a detector on a CUDA device"
and "forward a batch through it"; the search plan, the coordinate engine, the
trial-summary writer, the lock writer and the lock verifier all run for real):

1. `test_track_g_final_lock_carries_the_approved_unique_anchor_resolution` —
   Track G's final lock section is `UNIQUE_INHERITED_ANCHOR`, anchor
   `{head_lr: 1e-4}`, and the `learning_rate_multiplier` coordinate is present
   among the retained trials (i.e. it was actually searched, not blocked).
2. `test_track_r_final_lock_carries_the_approved_common_multiplier_resolution` —
   Track R's final lock section is `B_common_multiplier`, and the
   backbone:head ratio is 1:10 at every multiplier.
3. `test_the_pre_decision_diagnostic_cannot_be_mistaken_for_a_final_blocker`
   (parametrized G/R) — the exact stale-report scenario: proves
   `inherited_anchor_report` is explicitly scoped
   (`diagnostic_scope == "PRE_DECISION_STRUCTURAL"`), still shows
   `ambiguous: ["learning_rate"]` / `executable_under_full: false` (that part
   is legitimate and unchanged), and that this coexists with a track that
   really ran to completion (`trials_by_status.PASS > 0`, a real
   `winner_checkpoint_sha256`) and a lock that still verifies clean.
4. `test_deleting_the_pre_decision_diagnostic_does_not_change_lock_validity` —
   deleting `inherited_anchor_report` from both tracks does not change
   `verify_detector_config_lock`'s verdict, proving it is not a gating input.
5. `test_c8_track_configuration_reads_only_final_fields_not_the_diagnostic` —
   poisons `inherited_anchor_report` on a copy of the lock and asserts
   `c8.track_configuration` / `c8._track_parents` return the same values as on
   the unpoisoned lock.
6. `test_target_access_is_zero_throughout_the_finalized_lock` — invariant
   check on the finalized lock's `target_access` and
   `no_target_capability_proof`.
7. `test_reporting_metadata_never_enters_the_search_plan_identity` — see §2.

**No training is triggered by any of these**: `_StubTrainer` never touches
CUDA or a real dataset, and every new test either builds a `SearchPlan`
directly or drives the existing CPU-fixture-backed `scientific` sandbox
fixture that was already exercising the full lifecycle before this change.

**Broader regression**, exact command:

```
/…/testenv/bin/python3 -m pytest tests/c7 tests/pipeline -q -p no:cacheprovider
```

With this change: **34 failed, 1503 passed, 22 skipped** (310.81s).
`git stash` / `git stash pop` around a second run of the identical command,
against the unmodified starting HEAD `17a401651d96226c7e04933d98b1534f19e26605`
with no other change to the environment: **33 failed, 1496 passed, 22 skipped**
(269.48s). `comm` over the two sorted `FAILED` line sets shows the two runs
disagree on exactly **one** test:

- `tests/pipeline/test_adapters_integration.py::test_importing_the_pipeline_package_does_not_load_a_vendor_sdk_or_torch`
  — fails only in the *first* (this-change) run, because that run started
  before this session's throwaway virtualenv had `prism_fas` installed in
  editable mode (`pip install -e . --no-deps`, run between the two pytest
  invocations). It is an artifact of this session's environment setup order,
  not of the code change; both runs otherwise agree on every failing test id.

Every other failing test id is **identical between the two runs** — i.e.
pre-existing at the unmodified starting HEAD, reproduced with zero code change.
The set is dominated by tests that need infrastructure this laptop checkout
does not carry: a real Version-B checkout at the pinned path
(`test_checks_and_firewall.py::test_version_b_is_at_the_frozen_commit` and
siblings), the M2/M3A source-image pipeline over real frames
(`test_m2_m3a_contract.py`, 20 tests), a provisioned host bootstrap
(`test_bootstrap_host_interpreter.py`), the live orchestrator run over a fully
materialized repo (`test_orchestrator.py`), and one C3 live-replay archive
fixture (`test_adapters_c3_modes.py::test_live_binding_blocks_without_a_quota_snapshot`).
None of the 33 shared failures touch `tests/c7`, C7's or C8's adapters, the
search-plan module, or any file this fix changed.

**No test in `tests/c7` or `tests/pipeline` failed for a reason connected to
this change**, and all 8 new regression tests (§4 above) pass in both the
focused run and as part of this broad run.

---

## 5. C7 closure

```
Track G winner config SHA256:      97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd
Track G winner checkpoint SHA256:  08e4f10083990bfbb846761d8d63ac82e5bdadb2765ae194ce6ae992c680133e
Track G: 15/15 PASS, epoch 6, UNIQUE_INHERITED_ANCHOR, head_lr anchor 1e-4

Track R winner config SHA256:      d8537b257b84d394f0a43085bba2dc2fb1cafb2533ac199d172001df6d631110
Track R winner checkpoint SHA256:  8c81c1d16b066ab3a2e88d0dae0c43d182a09b751b9c6cb699a75f67c0d8c2be
Track R: 24/24 PASS, epoch 11, B_common_multiplier, backbone:head = 1e-5:1e-4 (1:10)

C6 training bank (DET):            32f6e0e129da277d0be76abc0758bc3d63ad16caba111efc801ac92de291f5b0
C6 selector identity:              27c20b71ff7c1d42dca1f7034f81bfd61400b5a17a2421fe823603c102e17ef3
C6 quality-threshold identity:     8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19
Source package identity:           08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9
C7 search-decision identity:       ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
LR decision identity:              16800cb4da6167d66ab34f1b444e794ff7ac6b96c3873fb8fcd9eb2a75207e58
target_access:                     0
metrics_from_trained_runs:         true (M9Trainer source_dev evaluation over the frozen
                                    C6 DET matched bank — not the engineering coordinate
                                    probe)
```

The first GPU C7 attempt (`3f18628511be7d022924b379db0599f5e4f8d87e`,
`recipe_text_cache.npz` missing) remains classified
`ENGINEERING_GLOBAL_INPUT_FAILURE`, `scientific_negative_result: false`,
`candidates_consumed: 0`, quarantined under `reports/evidence/quarantine/`, and
is untouched by this session — it is a distinct, negative engineering result,
not superseded evidence, and is not to be confused with the completed run
above.

**Closure verdict:**

```
C7_RETRAIN_REQUIRED       = NO
C7_SCIENTIFIC_RESULT_VALID = YES
```

The `inherited_anchor_report` finding is a reporting-clarity defect in
already-correct evidence, not a scientific inconsistency: no check, gate or
downstream consumer ever derived a decision from the ambiguous vocabulary it
carries, and the actual resolved state (`lr_interpretation`,
`lr_anchor_vector`, and now `lr_decision_resolution` /
`lr_decision_identity`) was always present and correct beside it.

---

## 6. Relationship to `reports/readiness/C7_GPU_SCIENTIFIC_HANDOFF.md`

That document is the pre-launch operator handoff for the run this audit
closes, and it is left as committed historical evidence — its "⛔ BLOCKED — do
not launch yet" banner describes the state before the `recipe_text_cache.npz`
recovery and the completed run; it is not rewritten here (git history and
prior milestone evidence are immutable per project policy). This document
supersedes it for CURRENT STATUS purposes: **C7 is closed, scientifically
valid, and this laptop did not touch its evidence.**

See `reports/readiness/C8_GPU_SCIENTIFIC_HANDOFF.md` for what runs next.
