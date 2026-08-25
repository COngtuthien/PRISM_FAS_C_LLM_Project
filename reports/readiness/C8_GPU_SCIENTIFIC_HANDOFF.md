# C8 GPU scientific handoff — the source-only experiment matrix

**This is a readiness audit and an execution handoff, not scientific
evidence.** C8 has not run scientifically. This document only establishes that
it is safe to authorize, and gives the exact command for when the user
authorizes it. **C8 WAS NOT RUN as part of producing this document.**

Audited at source HEAD: see §7 (git) of this session's final report.

---

## PRE-LAUNCH ENGINEERING DEFECT FOUND AND FIXED, BEFORE ANY C8 SCIENTIFIC RUN

The version of this document at HEAD `17d121ca82caaec5a46d181166197fb1f23fed29`
said

```
python train.py --profile full --from C8 --to C8 --preflight-only
```

was safe to run before authorization, to "confirm the three required inputs
beyond the C7 lock are present." **That statement was false.**
`train.py::_explicit` — the code path `--profile full ...` dispatches to —
parsed `--preflight-only` into `args.preflight_only` and then never read it:
it called `orchestrator.run(...)` unconditionally. The flag had real,
documented, working behavior in zero-argument mode (`_zero_argument` checks it
and returns *before* calling `run()` at all) and no effect whatsoever in the
explicit `--profile` mode this handoff's command uses. Nothing stopped that
exact command from reaching `C8Adapter.workflow()` and training a real
detector, subject only to whether C8's own preconditions happened to be
satisfied — which, per this same handoff, they were expected to be.

This was found and fixed, with regression tests, **before this handoff was
used to launch anything**. No C8 row ever ran under the broken command; no
detector trained; no target was accessed. See
`reports/readiness/C7_SCIENTIFIC_CLOSURE_AUDIT.md`'s git section and this
repository's commit history for the exact fix commit. The fix:

* threads `preflight_only` onto `AdapterRequest`, through
  `orchestrator.run(..., preflight_only=...)`, into every
  `EngineeringAdapter.run()` (C4-C13, which includes C8);
* stops that method before `workflow()` whenever `preflight_only` is set —
  after running the SAME `full_precondition_gate` (verify_c6_evidence,
  verify_detector_config_lock, checkpoint hash, accelerator) a real run would
  apply, never a second, looser check;
* makes `orchestrator.run()` return before `_write_reports`, the
  `MASTER_RUN_INDEX` row writer and `write_state` whenever `preflight_only` is
  set, so a preflight pass cannot stamp a `scientific_eligible=true` row, add a
  stage to `completed_stages`, or write anything under `reports/full/c8/` —
  the state-mutation risk the fix closes, not merely the training risk;
* is proven, at the `train.py` CLI boundary and against a real C7 lock built
  by actually running C7's scientific path (not a hand-written fixture), in
  `tests/pipeline/test_explicit_preflight_only.py`.

**The corrected, tested preflight command is unchanged in its spelling** —
`python train.py --profile full --from C8 --to C8 --preflight-only` — it is
the CODE BEHIND it that changed. §3 below gives the corrected, now-honest
description of what it does and does not do.

---

## SECOND PRE-LAUNCH ENGINEERING DEFECT: C8_PRELAUNCH_REQUIRED_INPUT_ROOT_DRIFT

**Found by the first real corrected, read-only C8 preflight, run on the GPU
host, before any C8 scientific row executed:**

```
python train.py --profile full --from C8 --to C8 --resume --preflight-only

C8 FULL_PRECONDITION_GATE BLOCKED
checks 6/7
Failed:
    c8_input_pretrained_weights: data/packages/pretrained is absent
```

The preflight itself worked correctly — it ran no workflow, wrote no
scientific state or artifacts, and honestly reported what it checked. What it
checked was wrong. `C8Adapter.required_inputs()` declared:

```
pretrained_weights  ->  data/packages/pretrained
source_packages     ->  data/packages
```

Neither is a real scientific root. `data/packages/pretrained` is a path
nothing in this codebase ever writes — the actual pinned SigLIP2/ConvNeXt
weights live under `sources.WEIGHT_ROOT = "weights"`, the same root
`C7Adapter.required_inputs()` already declared, and the same root C7's
completed GPU scientific run actually trained against. `data/packages` is the
PARENT of the real M3B source package
(`sources.SOURCE_PACKAGE_ROOT = "data/packages/prism_data_v1_m3b"`) — a
directory existing one level up proves nothing about the package itself, and
is in fact the exact class of defect `sources.py`'s own header comment
already documents as having been found and fixed once before for
`SOURCE_PACKAGE_ROOT` itself ("`SOURCE_PACKAGE_ROOT` was `data/packages` — the
parent directory..."). C8 had silently reintroduced it locally.

**Deeper gap, found by the same audit:** even with the roots corrected, C8's
precondition gate only ever checked *presence* of these paths — never the
SHA-verified pinned weights, the frozen recipe text cache's file SHA-256 and
re-derived semantic identity, the validated M3B package status, or the frozen
M7 recipe bank, all of which C7's `_scientific_prepare` already required
before its first trial through `sources.verify_detector_inputs`. A directory
merely existing was never proof any of that was true.

**Fix**, both parts reusing the canonical owning module rather than
re-spelling or re-implementing anything:

* `required_inputs()` for both C7 and C8 now import
  `sources.WEIGHT_ROOT`, `sources.SOURCE_PACKAGE_ROOT`,
  `sources.C5_CANDIDATES_ROOT`, `c6_evidence.C6_REPORTS` and
  `c7.SCIENTIFIC_CONFIG_LOCK_PATH` instead of hand-spelling the same strings —
  a fast presence-only first layer that structurally cannot drift between the
  two stages again, because both read the same module attribute.
* `C8Adapter.semantic_preconditions()` now ALSO calls
  `sources.verify_detector_inputs()` — the identical, SHA-verifying canonical
  check C7 already used — so the preflight gate genuinely validates the pinned
  weights, the text cache and the package, not merely their presence. There is
  no second, looser verifier anywhere in this path.
* C8 was also missing a `c5_candidates` required-input entirely (present in
  C7's declaration); it is now declared, using
  `sources.C5_CANDIDATES_ROOT`.

**Not fixed by creating `data/packages/pretrained`.** That directory is not
created anywhere in this change; doing so would have hidden the drift instead
of closing it.

This record is preserved, not erased, alongside the FIRST defect above — both
were found on real GPU-host or GPU-equivalent preflight runs before any C8
scientific row executed. See `tests/pipeline/test_c8_precondition_root_drift.py`
for the regression suite (path/import audit, real SHA-verification mechanics
against small fixture weights under a monkeypatched pin, and end-to-end
`orchestrator.run(..., preflight_only=True)` proof that a real failure from
the canonical verifier still BLOCKS and a real success still PASSES).

---

## 0. What C8 does, in one paragraph

C8 trains the frozen §18 source-only experiment matrix: 42 rows spanning
`C-G-RND` / `C-G-DET` / `C-G-LLM` (Track G) and `C-R-DET` / `C-R-LLM` /
`C-R-NOPROMPT` (Track R, the last being the mandatory PromptHead ablation),
each at its protocol (P1/P2/P3) and seed. Every row of a track trains at
**that track's one C7-frozen configuration** — never a per-arm or per-row
configuration — and differs only in which C6 matched bank supplies its
synthetic quarter (by arm), its protocol's source domains, and its seed.
Selection and calibration are source_dev only; cross-source comparison is
diagnostic; nothing in C8 reads or resolves a SiW-Mv2 target label.

---

## 1. C7 closure this handoff depends on

`reports/readiness/C7_SCIENTIFIC_CLOSURE_AUDIT.md` records the full audit.
Summary of what C8 will read:

```
Track G winner config SHA256:      97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd
Track G winner checkpoint SHA256:  08e4f10083990bfbb846761d8d63ac82e5bdadb2765ae194ce6ae992c680133e
Track R winner config SHA256:      d8537b257b84d394f0a43085bba2dc2fb1cafb2533ac199d172001df6d631110
Track R winner checkpoint SHA256:  8c81c1d16b066ab3a2e88d0dae0c43d182a09b751b9c6cb699a75f67c0d8c2be
C6 bank locks (RND/DET/LLM):        required by c8._track_parents / bank readers, all 3 present
C7 search-decision identity:        ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
target_access:                      0
C7_RETRAIN_REQUIRED:                NO
C7_SCIENTIFIC_RESULT_VALID:         YES
```

C8's own precondition gate (`C8Adapter.semantic_preconditions`) re-verifies
this independently, at launch time, on the GPU host's actual files — it does
not trust this document. It runs:

- `prism_fas.evaluation.c6_evidence.verify_c6_evidence` over
  `reports/full/c6` — the same three matched-bank closure C7 verified.
- `prism_fas.pipeline.adapters.c7.verify_detector_config_lock` over
  `reports/full/c7/DETECTOR_CONFIG_LOCK.json` — the identical, MODULE-LEVEL
  shared verifier C7 used to write the lock (`lock_verifier_shared_with: C8`
  in `docs/PROJECT_STATE.md`). There is no separate, laxer C8 verifier.
- `prism_fas.pipeline.adapters.sources.verify_detector_inputs` — added by the
  second defect fix above — the same canonical, SHA-verifying check C7's own
  `_scientific_prepare` already required: the validated M3B package
  (`data/packages/prism_data_v1_m3b`), the frozen M7 recipe bank, the pinned
  SigLIP2 tower and ConvNeXt V2 Atto weight by content SHA-256
  (`weights/...`), the frozen recipe text cache by file SHA-256 and
  re-derived semantic identity (`weights/recipe_text_cache.npz`), and the C5
  candidate tree (`runs/full/c5/scientific/candidates`).

If any of these fail, C8 blocks before training row 0. The reporting-only fix
in the closure audit does not change what the lock verifier checks (§1.3
there), so it will not fail on the reporting grounds the audit investigated.

---

## 2. What C8 may not do — confirmed from source, not asserted

- **Cannot use the quarantined first-attempt engineering failure.** That
  evidence lives under `reports/evidence/quarantine/`, carries
  `scientific_negative_result: false` / `candidates_consumed: 0`, and produced
  no `DETECTOR_CONFIG_LOCK.json` (`c7_first_gpu_attempt.detector_config_lock:
  none` in `docs/PROJECT_STATE.md`). C8's precondition gate requires a lock
  that *verifies*; a run that produced no lock cannot satisfy it, quarantined
  or not.
- **Cannot silently retrain or retune C7.** `c8.py::track_configuration` /
  `_detector_config_for_row` read the frozen `winner_config` and loss weights
  straight out of the C7 lock's per-track section — there is no code path in
  `c8.py` that calls C7's search machinery, `coordinate_search`, or
  `_scientific_finalize`. C8 has no writer for `DETECTOR_CONFIG_LOCK.json`.
- **Cannot reach a rehearsal/fixture detector under a scientific context.**
  `workflow()` dispatches `_engineering_workflow` vs `_scientific_workflow` on
  `context.is_scientific`, and `_run_scientific_row` calls
  `assert_fixture_permitted` before building anything — the defect this
  milestone's own audit found and fixed
  (`C8_FIXTURE_EXECUTOR_REACHABLE_UNDER_SCIENCE`,
  `docs/PROJECT_STATE.md::rehearsal_discovered_defects`) — is a structural
  RAISE, not a convention.
  `tests/pipeline/test_scientific_fixture_leakage.py` and
  `tests/pipeline/test_c8_scientific_path.py` hold this as a regression.
- **Cannot touch the target.** No C8 code path resolves `sources.target_roots`
  or a SiW-Mv2 label; `target_paths_resolved: 0` /
  `target_labels_resolved: 0` are written into every row manifest as a
  structural constant, not computed from an attempted access.
  `p3_cross_source: none` — a P3-ready row's held-out evaluation is deferred
  to C11, outside C8 entirely.

---

## 3. The C8-only GPU command

Run only after the user explicitly authorizes it:

```bash
git rev-parse HEAD                     # confirm the source HEAD you intend to run
git status --short                     # confirm a clean tree
python train.py --profile full --from C8 --to C8 --resume
```

`--resume` is identity-aware (L.11): a row already completed under matching
parent identities, config identity and content hash is skipped, never
retrained; the scheduler samples the true pending remainder from the frozen
42-row plan, not a re-sampled window.

**This command trains C8 only.** Do not append `--to C9` or run C9 in the
same invocation — `docs/PROJECT_STATE.md::c9` records C9 as
`still_blocked_by: DETECTOR_RELIABILITY_LOCK_C`
(`DETECTOR_BA_SEP_PROBE_PROTOCOL`, `DETECTOR_BA_SEP_EVIDENCE_VECTOR`,
`DETECTOR_BA_SEP_PROBE_SEEDS` remain `NEEDS_SCIENTIFIC_DECISION`, and per
`3f18628` detector reliability is a post-C8, pre-C9 barrier — it may not be
resolved from C8's own outcomes). C8 may finish; C9 stays blocked until that
decision is made, separately and not from a C8 result.

Before running, confirm on the GPU host that the three required inputs beyond
the C7 lock are present. **This is now genuinely read-only** (see the defect
record above): it verifies C6 evidence, verifies the C7 lock with C7's own
`verify_detector_config_lock`, checks the winner checkpoints' bytes and hashes,
probes for an accelerator, and stops — `workflow()` is never called, so this
cannot train, step an optimizer, write a checkpoint, mark a row complete, or
write anything to `reports/full/c8/`, `runs/full/c8/`,
`state/PIPELINE_STATE.json` or `state/MASTER_RUN_INDEX.json`:

```bash
python train.py --profile full --from C8 --to C8 --preflight-only
```

It prints `preflight  PASS` (every precondition satisfied, nothing executed)
or `preflight  BLOCKED` (names the exact unsatisfied precondition, exactly as
a real run's gate would) and exits 0 or 2 accordingly. A BLOCKED verdict here
means the real run in the command above it would also block on the same
input — fix that input and re-run preflight before spending GPU time.

---

## 4. STOP conditions specific to this launch

Stop and report rather than proceeding if, at launch time on the GPU host:

- `verify_detector_config_lock` reports `valid: false` for any check —
  report the exact `check_id`, do not patch around it;
- `verify_c6_evidence` reports the closure invalid;
- the C7 lock's `winner_checkpoint_sha256` for either track does not match
  the bytes on disk (checkpoint corruption/drift);
- the scheduler proposes training more than 42 rows, or a row whose
  `experiment_id` is not one of `C-G-RND`, `C-G-DET`, `C-G-LLM`, `C-R-DET`,
  `C-R-LLM`, `C-R-NOPROMPT`;
- any row's manifest reports `target_paths_resolved > 0` or
  `target_labels_resolved > 0`;
- Version B's HEAD/tag/clean state has drifted from
  `7799f7decd35db6987ce4578824e5bd8d9eab4ae` /
  `m10-blind-evaluation-checkpoint`.

---

## 5. Verdict

```
C8 READY:  YES
BLOCKERS:  none identified from source; final confirmation is the GPU host's
           own --preflight-only run (now genuinely read-only, and now checking
           the real canonical roots and the real SHA-verified inputs — see
           both defect records above) and the live semantic_preconditions
           gate, which this document does not substitute for.

PRE-LAUNCH ENGINEERING DEFECT 1 (EXPLICIT_PREFLIGHT_ONLY_NOT_HONORED):
           found and fixed before any C8 scientific run. --preflight-only
           under the explicit --profile path did not stop execution. Fixed
           with regression tests. No C8 row, checkpoint, optimizer step or
           target access occurred under the broken command.

PRE-LAUNCH ENGINEERING DEFECT 2 (C8_PRELAUNCH_REQUIRED_INPUT_ROOT_DRIFT):
           found by the first corrected, genuinely read-only preflight run on
           the real GPU host. C8's required inputs named
           data/packages/pretrained (nothing ever writes this path) and
           data/packages (the M3B package's parent, not the package) instead
           of the canonical weights and data/packages/prism_data_v1_m3b roots
           C7 already trained against. Fixed to import the canonical roots
           and to run sources.verify_detector_inputs (the SHA-verifying
           check) as part of the gate, with regression tests. No C8 row,
           checkpoint, optimizer step or target access occurred under the
           drifted command; both preflight runs it produced (BLOCKED, then
           the correct verdict once the host supplies the real assets) are
           preserved as engineering evidence, not erased.

C8 WAS NOT RUN as part of this audit or as part of finding/fixing either
defect.
```
