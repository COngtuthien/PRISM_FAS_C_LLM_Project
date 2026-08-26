# C9_DETECTOR_BA_SEP_OPTION1_V2 — runner integration fix

**DISCOVERED BEFORE ANY BA_sep VALUE WAS EVER OBSERVED.** An audit of the
runner CLI committed in the previous task
(`reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md`) found
two integration defects between that CLI and the already-frozen V2 protocol.
**No real bind had yet occurred. No real BA_sep had yet occurred.**
`--execute` had remained deliberately unwired (`run_scientific_probe`
raised `NotImplementedError`). This task fixes both defects. **It does not
change the V2 scientific protocol.**

---

## 1. The two defects

### 1.1 Per-arm binding contradicted the joint balancing rule

The committed `--bind-only --arm {RND,DET,LLM}` resolved and wrote a
binding for ONE arm at a time. The frozen protocol's balancing rule
(`class_balancing_rule.count_rule` in
`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml`, implemented in
`synthetic_real_probe.balance_classes`) is:

```
N = min(real_spoof_available, RND_synthetic_available,
        DET_synthetic_available, LLM_synthetic_available)
```

Binding one arm at a time necessarily supplied `[]` for the other two arms'
synthetic pools (there was no joint call that had all three at once), which
forces `N = min(real, this_arm, 0, 0) = 0` for every cell. This was not a
scientific decision — it is an implementation that could never satisfy the
protocol it claimed to implement. No bind had ever succeeded against real
data (this laptop has never had the GPU artifacts to try), so no invalid
binding was ever produced; the defect was caught in review before first use.

### 1.2 `--execute` was not real code

The committed `_execute` routed to `synthetic_real_probe.run_scientific_probe`,
which deliberately raised `NotImplementedError` — a correct and honest state
for THAT task (it explicitly scoped itself to freezing and implementing
protocol mechanics, not executing them), but it meant the CLI was not yet a
scientific runner at all.

## 2. The fix

### 2.1 Joint architecture — no `--arm` on any mode

`python -m prism_fas.evaluation.synthetic_real_probe_runner --repo . {--preflight-only|--bind-only|--execute}`.
No mode takes `--arm`. The synthetic-vs-real reliability test is one
preregistered three-arm experiment, bound and executed as one unit:

- `synthetic_real_probe.build_checkpoint_binding(repo)` resolves all 15
  checkpoints (5 per arm × 3 arms) atomically via
  `resolve_all_checkpoint_sets`, raising unless every one resolves.
- `synthetic_real_probe.build_population_plan(repo, protocol=...)` resolves
  the real population ONCE
  (`resolve_joint_populations`/`resolve_real_spoof_population`) and each
  arm's synthetic population once, then for every
  `(probe_seed, source_domain, split)` cell calls `balance_report` with the
  REAL three-arm mapping `{"RND": ..., "DET": ..., "LLM": ...}` — never a
  per-arm call with the other two arms filled with `[]`.

### 2.2 Zero-sample cells fail closed

`build_population_plan` raises `SyntheticRealProbeError` if any required
cell resolves `N == 0`, and additionally asserts, per cell, that every arm's
synthetic selection count equals the real selection count (the 1:1 balance
the protocol requires). A leakage audit
(`_population_plan_leakage_audit`) re-checks, across every cell built for a
probe seed, that no `stable_group_identity` appears in both the train and
validation unions — on top of the per-population `verify_group_safe_split`
checks already performed while building each split.

### 2.3 Two global, atomic artifacts

`--bind-only` writes exactly two files, under
`reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`:

- `C9_BA_SEP_EXECUTION_BINDING.json` — all 15 checkpoint bindings (arm,
  seed, row_id, run_identity, config_identity, checkpoint_relative_path,
  checkpoint_sha256, decision_graph_hash, decision_logit_name), the source
  package identity, all three C6 bank identities, and a
  `checkpoint_binding_identity_sha256`. No performance metric appears in it.
- `C9_BA_SEP_POPULATION_PLAN.json` — the exact preselected `sample_identity`
  / `stable_group_identity` lists for every `(probe_seed, source_domain,
  split)` cell, for the real population and each arm's synthetic
  population, plus pre/post-balance counts, unique group counts and the
  leakage audit, bound to a `population_plan_identity_sha256`.

Both are built fully in memory first; only after every check passes are
they written, atomically (`prism_fas.pipeline.state.atomic_write_json`).
Nothing partial is ever written — no per-arm `BINDING_{ARM}.json` files
exist anywhere in this codebase any more. If matching artifacts already
exist (identical identity), they are verified and reused; if an existing
artifact carries a DIFFERENT identity, binding is refused rather than
silently overwriting a prior preregistration.

### 2.4 Strengthened production preflight

`--preflight-only` now passes only if ALL of: the V2 protocol resolves, the
source package and all three C6 arm banks resolve, all 15 C8 checkpoints
resolve and hash-verify, the real/synthetic `source_record_id` group-identity
mapping resolves end to end (`resolve_joint_populations`), and the protocol's
own target firewall fields read `target_access: 0`. Still strictly
read-only — no function it calls fits a probe, loads a checkpoint's weights,
or writes a file. On this development laptop (no M3B package, no
`runs/full/c8/`) it correctly reports `BLOCKED`.

### 2.5 `--execute` is now real code

`synthetic_real_probe.execute_joint_probe(repo, checkpoint_binding=...,
population_plan=...)` re-verifies both artifacts are bound to the currently
active protocol identity and to their own recorded identity hash, then for
every arm strict-loads all five bound checkpoints
(`construct_row_trainer` — the exact C8 row-construction path: the row's
own `SourceRow`, the frozen C7 Track-G config lock, the arm's frozen C6
bank, `pipeline.adapters.c8._detector_config_for_row`,
`detector.trainer.M9Trainer` pointed at the row's own existing run
directory, then `detector.checkpoint.load_checkpoint`/`apply_checkpoint`,
strict, identity-checked), forwards every required real and synthetic
sample through each checkpoint (`forward_evidence_for_records`, reusing
`forward_checkpoint_evidence`/`extract_evidence` exactly — no second
evidence rule), averages evidence across the five checkpoints per arm
(`average_checkpoint_evidence`), fits the frozen linear probe per
`(arm, probe_seed)` on the prebound population plan
(`compute_ba_sep_for_seed`, unchanged), aggregates
(`aggregate_ba_sep`, unchanged) and applies the frozen all-arm hard verdict
rule (`hard_verdict`, unchanged). `construct_row_trainer` never calls
`run_source_only_flow`, `.backward()`, `.optimizer.step()`, or `.save()` —
verified by a static source check, not just by convention.

A checkpoint whose bytes on disk no longer match its bound SHA-256, a
sample missing evidence from any one of the five checkpoints, or an
artifact bound to a stale protocol identity all fail closed (raise) before
any BA_sep is computed. A genuine scientific `FAILED` verdict (any arm's
`BA_sep > 0.75`) is written honestly to the same five result artifacts a
`PASS` would be — it is a real result, not a runner error, and the CLI
distinguishes it with its own exit code (1) from a `BLOCKED` precondition
failure (2).

### 2.6 Scientific result artifacts (written only by a real `--execute`)

`BA_SEP_RESULT.json`, `BA_SEP_PER_SEED.json`, `BA_SEP_PROBE_PARAMETERS.json`,
`BA_SEP_EVIDENCE_MANIFEST.json`, `SYNTHETIC_VS_REAL_SPOOF_PROBE_VERDICT.json`
— all under the same reliability directory, all atomic, all binding the
protocol identity, the checkpoint binding identity, the population plan
identity, the source package identity, all three C6 bank identities, every
bound checkpoint's SHA-256, the code commit and `target_access: 0`. None of
these files exists in this repository — `--execute` refuses on every host
today, including this one, because no `--bind-only` has ever succeeded here.

## 3. What did NOT change

- The V2 scientific protocol itself
  (`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml`) —
  byte-for-byte identical; its identity
  (`720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`) is
  unchanged, confirmed by regression.
- No V3 protocol was created. This is an implementation fix, not a
  scientific correction — §23 of the task explicitly required stopping and
  reporting `NEEDS_SCIENTIFIC_DECISION` rather than silently editing V2 if
  a result-affecting field change had turned out to be necessary; it was
  not.
- The evidence vector, checkpoint policy, probe seeds, linear-probe
  hyperparameters, threshold, BA aggregation, ceiling and hard verdict rule
  — all unchanged, all reused exactly as frozen.
- `C_H4_SUPPORT_RULE` in `detector_reliability.py` — untouched; every
  verdict artifact carries it alongside the hard-gate verdict so a PASS is
  never read as C-H4 support.

## 4. A latent bug found and fixed while making this correction

`resolve_checkpoint_set` constructed the bound checkpoint's relative path by
hand (`row_directory(...) / "checkpoint.pt"`), but a real C8 row manifest
records its checkpoint under `run_root/checkpoints/{kind}.pt` where `kind`
is `"best"` or `"last"` (`M9Trainer.checkpoint_path`,
`c8.py::_run_scientific_row`'s own `checkpoint = trainer.checkpoint_path("best")`
/`"last"` fallback). The hand-built path was wrong and would have pointed at
a file that does not exist. `resolve_checkpoint_set` now reads the
checkpoint's actual relative path and kind directly off the same
already-hash-verified manifest `source_evidence.load_row_evidence` opened
(`RowEvidence` does not carry those two fields, so this is completing what
that shared contract deliberately leaves out — never a second
implementation of it), and additionally refuses (fails closed) any row
whose manifest names a decision logit other than `global_logit_G`.

`checkpoint_binding_identity`/`population_plan_identity` also had a
self-reference bug: neither excluded its own `..._identity_sha256` field
from the material it hashes, so re-verifying an artifact ALREADY WRITTEN TO
DISK (which now contains that field) computed a different hash than the one
originally stored — every legitimate re-verification (`execute_joint_probe`,
a repeated `--bind-only`) would have wrongly reported a mismatch. Both
functions now exclude their own field, matching the pattern
`detector_reliability.protocol_identity` already used for `frozen_on`/
`approved_by`/etc.

Both were caught by this task's own fixture-based tests before being
committed — no real artifact was ever affected, since no real bind or
execute has ever run on any host.

## 5. Tests

`tests/pipeline/test_c9_ba_sep_option1_protocol.py` — 44 tests (V1/shared
mechanics), one test updated (`run_scientific_probe`'s retirement now
raises `SyntheticRealProbeError`, not `NotImplementedError`).

`tests/pipeline/test_c9_ba_sep_option1_v2_runner.py` — rewritten CLI/joint
sections (90 tests total in the file): V1 untouched; V2 supersession and
identity (unchanged from the prior task); group-identity resolution;
split/selection separation; group-safety; balance-report group counts; the
hard verdict rule; joint checkpoint-binding and population-plan mechanics
(atomic, all-three-arms, zero-cell fail-closed, leakage audit, real-subset
reuse across arms); the CLI's three joint modes (`--preflight-only`,
`--bind-only`, `--execute` — no `--arm` anywhere, strengthened preflight,
atomic two-artifact bind, refuse-on-mismatch reuse contract); real
execution orchestration exercised end to end with monkeypatched
`construct_row_trainer`/`forward_evidence_for_records` (a PASS-verdict
fixture with indistinguishable evidence, a FAIL-verdict fixture with
cleanly separable evidence, missing-checkpoint-evidence fail-closed,
identity mismatch refusal, C-H4 separation); and source-level safety checks
proving `construct_row_trainer` never calls a training method and always
sets eval mode.

150 tests pass across the three C9-scoped files
(`test_c9_ba_sep_option1_protocol.py`, `test_c9_ba_sep_option1_v2_runner.py`,
`test_c9_scientific_evidence.py`). Broad regression (`tests/c7 tests/pipeline`):
33 failed / 1663 passed / 22 skipped — the exact, unchanged pre-existing
baseline (verified against the same 33 failures recorded before this task).

## 6. What this fix does NOT do

- Does not compute a real BA_sep value. `--execute` on this laptop remains
  `BLOCKED` (no `runs/full/c8/`, no M3B package) — verified: `--preflight-only`,
  `--bind-only` and `--execute` were each run once against the real repo as
  part of this task and each exited `BLOCKED`, wrote no file, and modified
  no state.
- Does not resolve any of the nine `REQUIRED_DETECTOR_RELIABILITY_TESTS`.
- Does not create `DETECTOR_RELIABILITY_LOCK_C.json`.
- Does not run C9, C10, C11, C12 or C13.
- Does not access target data. `target_access = 0` on every path, including
  every artifact this fix's real code would write.
- Does not retrain C7 or any C8 row, add `C-R-RND`, or alter any C7 winner.

**C9 remains correctly `BLOCKED_PENDING_DETECTOR_RELIABILITY_SCIENTIFIC_DECISION`
after this task.** Fixing the runner's integration with the already-frozen
protocol neither computes nor fabricates a pass for
`detector_reliability.verify_lock()`.
