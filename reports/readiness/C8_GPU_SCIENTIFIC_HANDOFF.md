# C8 GPU scientific handoff — the source-only experiment matrix

**This is a readiness audit and an execution handoff, not scientific
evidence.** C8 has not run scientifically. This document only establishes that
it is safe to authorize, and gives the exact command for when the user
authorizes it. **C8 WAS NOT RUN as part of producing this document.**

Audited at source HEAD: see §7 (git) of this session's final report.

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

If either fails, C8 blocks before training row 0. The reporting-only fix in
the closure audit does not change what this verifier checks (§1.3 there), so
it will not fail on the reporting grounds the audit investigated.

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
the C7 lock are present:

```bash
python train.py --profile full --from C8 --to C8 --preflight-only
```

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
           own --preflight-only run and the live semantic_preconditions gate,
           which this document does not substitute for.
C8 WAS NOT RUN as part of this audit.
```
