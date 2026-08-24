# C7 GPU scientific handoff — the first real detector search

**This is an execution handoff, not scientific evidence.** No C7 scientific
trial has ever executed, no detector has been trained, and
`reports/full/c7/DETECTOR_CONFIG_LOCK.json` does not exist anywhere.

> ## ⛔ BLOCKED — do not launch yet
>
> The first real GPU C7 attempt failed on a missing frozen artifact, and it is
> still missing. **`recipe_text_cache.npz` must be restored to the GPU host
> before this run can start.** It cannot be rebuilt. §2A has the recovery
> procedure, and §6 now blocks on it automatically.
>
> The invalid Track-G state that attempt left behind must also be quarantined
> first — §9A — or `--resume` will re-report a closed envelope forever.

Read the STOP rules before launching. C7 is a bounded one-pass search: the value
of the whole design is that the envelope cannot be widened after a result is
seen, so almost every failure mode below ends in "preserve and stop" rather than
"try something else".

---

## 0. What is being run, in one paragraph

C7 closes the §15.2.2 detector/loss envelope. It runs **one bounded coordinate
pass per track** — Track G and Track R — both training against the **frozen C6
DET matched bank**, and freezes one detector configuration per track into a
single `DETECTOR_CONFIG_LOCK.json`. C8 later trains all three Track-G generator
arms (RND, DET, LLM) at the Track-G configuration and both Track-R arms (DET,
LLM) at the Track-R one. There is no per-arm search, and there never will be:
that is the confound the design exists to remove.

---

## 1. Heads

| | |
|---|---|
| **Scientific implementation head** | `4d4bdad74dae4082535f604161234f9f9f08e737` |
| **Checkout head** | the tip of `portable-one-command-full-run` — call it `HANDOFF_HEAD` |
| Branch | `portable-one-command-full-run` |

`HANDOFF_HEAD` is the commit that added this file, and a file cannot contain its
own hash. Resolve it on the host rather than trusting a transcribed SHA:

```bash
HANDOFF_HEAD=$(git log -1 --format=%H -- reports/readiness/C7_GPU_SCIENTIFIC_HANDOFF.md)
echo "$HANDOFF_HEAD"
```

Everything between `4d4bdad74dae4082535f604161234f9f9f08e737` and `HANDOFF_HEAD` is documentation
and evidence only — no file under `src/`, `configs/`, `train.py`, `bootstrap.py`
or `assets/` differs. **Verify that before running anything:**

```bash
git diff --stat 4d4bdad74dae4082535f604161234f9f9f08e737 "$HANDOFF_HEAD" -- src train.py bootstrap.py configs assets
```

That must print nothing. If it prints anything, the scientific implementation
moved after `4d4bdad74dae4082535f604161234f9f9f08e737` and this handoff no longer describes what
would run — STOP and report.

---

## 2. The frozen scientific contract

### The search population

```
C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET          FROZEN
decision record   configs/search/c7_source_search_decision.yaml
decision identity ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
timing            BEFORE_FIRST_C7_SCIENTIFIC_TRIAL
```

§15.2.2 freezes the envelope and never says which of C6's three matched banks
supplies the synthetic quarter of the batch during the search. DET was chosen
explicitly: it is the structured non-LLM control and the only generator arm
primary in **both** tracks, so one non-treatment anchor serves both searches.
Tuning on LLM would have handed the proposed treatment an advantage over its own
controls. v1.5 does not choose DET; this is a recorded closure decision.

RND and LLM are **refused** as a search population. Their banks are still
verified — C8 trains on all three — but no RND or LLM candidate byte enters C7
training, and no RND or LLM detector performance is computed or read before the
lock closes.

### The learning-rate decision

```
lr decision record    configs/search/lr_anchor_decision.yaml   (unchanged bytes)
decision identity     16800cb4da6167d66ab34f1b444e794ff7ac6b96c3873fb8fcd9eb2a75207e58
superseded identity   7ef3492263507d4399828089bbe1af79438bc892e50c8ad732585c1d40c8397c
supersession record   reports/handoff/LR_ANCHOR_DECISION_CORRECTION.json
```

The identity moved because an implementation defect was corrected at `4d4bdad74dae4082535f604161234f9f9f08e737`,
**not** because the approved decision changed. `UNIQUE_INHERITED_ANCHOR` was
being executed as "the learning-rate coordinate is not searched", which omitted
Track G's learning rate — the first coordinate of the frozen order — and left
Track G with 12 trials instead of 15. The approved YAML was always correct and
no byte of it changed; only the code that read it did. Zero scientific trials
had run, so nothing needed regenerating. Full account in the supersession record.

If you are comparing against an older handoff or an older search state, this is
the identity that legitimately differs.

---

## 2A. The frozen recipe text cache — the current blocker

The PromptHead needs text embeddings for the 128 frozen M7 recipe descriptions.
They are an **uploaded artifact**, not a runtime computation:

```
file                   recipe_text_cache.npz
file sha256            bb7d3fb4b82ad6ac89ebb06eeac9eb679e2fbb3bab500112cd1e304c187683aa
cache identity         10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141
pin                    configs/models/m9_detector.yaml  (model.prompt)
historical location    /vol/models/pretrained/m9/recipe_text_cache.npz
```

**Do not rebuild it.** Encoding the 128 descriptions is deterministic within one
environment but *not* bit-identical across torch/transformers builds, or across
CPU and GPU. A rebuild on the RTX 5090 under the current environment would
produce a different content identity for the same science — which is precisely
what the identity pin exists to detect. Regenerating it does not solve this
problem; it hides it.

### Locate it

Search approved storage first — the Modal volume, the M9 artifact store, any
backup of the historical bundle:

```bash
find / -name 'recipe_text_cache.npz' -type f 2>/dev/null
```

### Accept a candidate only if BOTH match

```bash
sha256sum <candidate>
# must be bb7d3fb4b82ad6ac89ebb06eeac9eb679e2fbb3bab500112cd1e304c187683aa
```

```bash
cp <candidate> weights/recipe_text_cache.npz

/usr/bin/python3 - <<'PY'
from pathlib import Path
from prism_fas.detector.heads import resolve_recipe_text_cache
cache = resolve_recipe_text_cache(Path("weights"))
print("identity:", cache.identity)
print("recipes :", cache.count, "dim:", cache.dim)
assert cache.identity == "10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141"
print("OK")
PY
```

The file SHA proves the bytes are the frozen bytes. The identity check proves the
contents mean what the pin says — `read_recipe_text_cache` recomputes the
identity from the stored binding and the content identity from the embeddings, so
an edited cache cannot claim an identity it does not have. A file could satisfy
either alone; the frozen artifact satisfies both.

`weights/recipe_text_cache.npz` is the preferred destination.
`weights/m9/` and `weights/pretrained/m9/` are also searched.

### If the exact bytes cannot be found

**STOP. `HANDOFF = BLOCKED_ON_FROZEN_ARTIFACT`.** Do not mint a replacement, do
not rebuild, do not proceed with PromptHead disabled. Report that the original
frozen artifact must be recovered from the historical M9 artifact store or a
backup — that is a retrieval problem, not a modelling decision.

---

## 3. The declared cost — read this before launching

Two counts, and they are not the same number.

```
                 logical      unique
                 occurrences  configurations
Track G          15           11
Track R          24           17
                 --           --
total            39           28

per trial        35 epochs x 45 optimizer steps
compute          28 x 35 x 45  =  44,100 optimizer steps

effective batch (G5)   12 real-live + 12 real-spoof + 8 synthetic  =  32
```

A coordinate pass evaluates each coordinate's candidates while the others sit at
the current best. Whenever the anchor wins a coordinate, the anchor
**configuration** recurs at the next one — same canonical config SHA, different
search position. So 39 logical occurrences resolve to 28 distinct detector
trainings, and an occurrence of an already-trained configuration reuses that
run's evidence rather than retraining it.

An earlier version of this handoff quoted **61,425** steps. That is 39 × 35 × 45,
which counts every occurrence as a separate training. It is the right number for
"what if nothing were reused" and the wrong one for the compute estimate. The
honest figure is **44,100**; both are printed by the run.

Track G tunes five coordinates and Track R eight because their active loss sets
genuinely differ: Track G instantiates no ConvNeXt, no regions and no PromptHead,
so `lambda_local`, `lambda_MIL` and `lambda_P` are NOT APPLICABLE there, and
manifold-OFF Track R has no `L_real`/`L_out`/`L_clean` to weight.

Every trial runs the **full** frozen M9 schedule. Do not reduce epochs, steps,
batch composition or the number of candidates to fit the machine — L.12 forbids
shrinking a scientific budget to fit hardware, and a shortened schedule would
make the result a measurement of something else.

The fairness invariant is *one frozen configuration within a track*, not one
identical numeric vector across two different architectures.

**The run prints its own declared counts before the first trial.** Treat the
numbers above as the expectation, and the run's own
`C7_SCIENTIFIC_SEARCH_PLAN.json` as the authority. If they disagree — STOP and
report; something moved that should not have.

---

## 4. GPU host

```
repo    /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project
python  /usr/bin/python3
```

Do not rebuild `.venv`. Do not `pip install` anything. `train.py` bootstraps and
reuses the project environment itself.

---

## 5. Sync

```bash
cd /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project

git fetch origin
git checkout portable-one-command-full-run
git reset --hard origin/portable-one-command-full-run

HANDOFF_HEAD=$(git log -1 --format=%H -- reports/readiness/C7_GPU_SCIENTIFIC_HANDOFF.md)
git rev-parse HEAD
git status --short -- src train.py bootstrap.py configs assets
```

`git rev-parse HEAD` must equal `$HANDOFF_HEAD` — if the branch has moved on
since this handoff was written, stop and get a fresh one rather than running an
implementation this document does not describe. `git status --short` over that
scope must print **nothing**. Do not run if the scientific code/config tree is
dirty — a local edit under `src/` or `configs/` would silently change what is
being frozen.

---

## 6. Precheck

The run performs all of this itself and BLOCKS if any part fails, so this section
is what to expect rather than a script to run by hand. Path existence is never
sufficient anywhere below.

| # | Condition | How it is checked |
|---|---|---|
| 1 | code head is `HANDOFF_HEAD` | `git rev-parse HEAD` vs the resolved SHA above |
| 2 | scoped tree clean | `git status --short -- src train.py bootstrap.py configs assets` |
| 3 | real NVIDIA GPU present | `_scientific_device()` requires CUDA and refuses CPU outright |
| 4 | **C6 closure verifies strictly** | `prism_fas.evaluation.c6_evidence.verify_c6_evidence` — the canonical strict verifier, run as a C7 semantic precondition |
| 5 | C7 inputs resolve | `sources.verify_detector_inputs` |
| 5a | **frozen recipe text cache** verifies by file SHA-256 **and** re-derived identity | `sources._frozen_recipe_text_cache`, inside the same verifier — this is what was missing on the first attempt |
| 6 | search decision resolves DET / FROZEN | `prism_fas.search.c7_decision.load_decision` |
| 7 | `target_access = 0` | no target path, label or metric is reachable from any C7 code path |

Required inputs, all of which must be present **and valid**:

```
weights/                                 pinned SigLIP2 + ConvNeXt V2 Atto, SHA-verified
weights/recipe_text_cache.npz            the frozen PromptHead text cache  <- §2A
data/packages/prism_data_v1_m3b          M3B source package, status validated
reports/full/c6                          the frozen C6 closure
runs/full/c5/scientific/candidates       the rendered candidate bytes the banks address
```

The text cache is a `required_for_gpu_science` asset in the portable inventory
now, so the preflight names it in the missing-item list rather than letting the
run discover it 15 candidates in.

### Expected C6 identities — verify, never rewrite

```
selected profile              NOMINAL
quality threshold identity    8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19
selector identity             27c20b71ff7c1d42dca1f7034f81bfd61400b5a17a2421fe823603c102e17ef3

selected_set_sha256
  RND   1a5be34ad5104b9c8fe18f90778e86b6710180e757ea066c4f931565928d50ba
  DET   32f6e0e129da277d0be76abc0758bc3d63ad16caba111efc801ac92de291f5b0
  LLM   b7edd00b7ea87e558814b96955452f3bf4bdbeda0da4374461b4ed4ddd29525b
```

If any identity differs: **STOP.** Do not regenerate, repair or re-run C6. A
differing identity means the bank on disk is not the bank C6 froze, and that is
a fact to investigate, not a state to fix automatically.

The DET selected-set digest is bound into the search-plan identity and into every
trial's parent set, so a moved DET bank invalidates resume by construction rather
than by convention.

---

## 7. The run

```bash
set -o pipefail

PYTHONUNBUFFERED=1 /usr/bin/python3 -u train.py \
  --profile full \
  --resume \
  --from C7 \
  --to C7 \
  2>&1 | tee gpu_c7_${HANDOFF_HEAD:0:7}.log
```

**C7 only. Never add C8 to this invocation.** Stop after C7 whatever the outcome
— PASS, FAIL or BLOCKED. The one thing standing between C7 PASS and C8 is the
audit of the C7 lock; detector reliability is a *later* barrier and does not
gate C8. See §13 for the sequencing.

---

## 8. Expected sequence

```
VERIFY_C6_EVIDENCE          strict C6 closure + inputs + pinned weights
SCIENTIFIC_SOURCE_SEARCH    plan (identities + declared counts printed here)
                            Track G bounded pass   15 declared trials
                            Track R bounded pass   24 declared trials
FINALIZE_DETECTOR_CONFIG    one winner per track
VERIFY_CONFIG_LOCK          strict verification of what was just written
```

Both passes are DET-anchored. Governing artifact:

```
reports/full/c7/DETECTOR_CONFIG_LOCK.json
```

with exactly `tracks.G` and `tracks.R`, one frozen winner each.

Per-track scientific artifacts and state:

```
reports/full/c7/C7_SCIENTIFIC_INPUTS.json
reports/full/c7/C7_SCIENTIFIC_SEARCH_PLAN.json
reports/full/c7/C7_SCIENTIFIC_SOURCE_SEARCH_G.json
reports/full/c7/C7_SCIENTIFIC_SOURCE_SEARCH_R.json
reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_G.json
reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_R.json
runs/full/c7/scientific/trial_<config_sha[:16]>/C7_TRIAL_SUMMARY.json
```

Every trial writes a summary — PASS, FAIL and DIVERGED alike. §15.2.2 retains
invalid and divergent trials; a trial with no artifact would be
indistinguishable from one that never ran.

---

## 9. Resume

If SSH drops, the terminal dies or the process is killed:

```bash
# the IDENTICAL command, unchanged
PYTHONUNBUFFERED=1 /usr/bin/python3 -u train.py \
  --profile full --resume --from C7 --to C7 \
  2>&1 | tee -a gpu_c7_${HANDOFF_HEAD:0:7}.log
```

- **Completed valid trials are reused** by config identity and are not retrained.
- **An interrupted pass** resumes at the trial that stopped; only genuinely
  incomplete identical units execute.
- **A completed bounded pass is returned, not re-walked.** Re-running a finished
  search costs zero GPU trials.
- No coordinate is revisited and no second pass ever runs.

## 9A. Quarantining the first attempt's invalid state

The first attempt left `reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_G.json` with
`status: COMPLETED`, 15 FAIL rows and zero finite-valid trials. Under `--resume`
that is a **closed envelope**: the next run will re-raise `EnvelopeExhausted`
immediately and train nothing, forever.

It is not a scientific result — every row failed on the missing text cache before
training — but it is evidence about how the run failed, so it is preserved rather
than deleted.

```bash
# inspect first, and READ the output
/usr/bin/python3 scripts/recover_c7_invalid_search_state.py --track G

# only if it reports eligible
/usr/bin/python3 scripts/recover_c7_invalid_search_state.py --track G --apply
```

The script proves eligibility before touching anything. It **refuses** if any row
is finite-valid or PASS, if rows record differing or unrecognised causes, if the
search-plan identity does not match, if a `DETECTOR_CONFIG_LOCK` exists, or if
the trial artifacts show real training progress — checkpoints, completed stages
or selection metrics. It reads those artifacts rather than trusting the failure
message, so if training actually got further than the message suggests, it says
so and stops.

On `--apply` it copies the state, every trial summary and any `gpu_c7_*.log` into
`reports/evidence/quarantine/` with SHA-256 for each, writes a
`RECOVERY_RECORD.json` classifying the event as
`ENGINEERING_GLOBAL_INPUT_FAILURE` with `candidates_consumed: 0`, and then clears
**only** the active state file.

It does not touch the DET search arm, the §15.2.2 envelope, the LR decision, C6
or C5. This is not a result-driven restart: there was no configuration-specific
scientific result to restart from.

Send the `RECOVERY_RECORD.json` back with the run.

---

### `SEARCH_STATE_IDENTITY_MISMATCH`

If the run reports this, a bound frozen input moved — the C6 DET bank, the search
decision, the LR decision or the source package — and the recorded state belongs
to a different envelope. The run fails closed and writes nothing.

**STOP.** Do not delete the search state. Do not start a fresh pass. Do not move
or regenerate the C6 bank. Report which identity differs; the invalidation
subtree has to be worked out deliberately, not by clearing state until the run
proceeds.

---

## 10. Failure classification

### A0 — global dependency block (new)

A pinned artifact is missing or wrong, a frozen identity moved, the source
package or C6 bank vanished. The run reports
`c7_track_<x>_global_dependency_available: false` with
`reason_code: GLOBAL_DEPENDENCY_UNAVAILABLE`, names the dependency, and records
`scientific_envelope_exhausted: false` and `candidates_consumed: 0`.

**No candidate is consumed and the envelope is NOT exhausted.** This is a fact
about the host, true of every configuration equally. Fix the input and re-run;
nothing scientific happened.

This class exists because the first attempt had no such class: every candidate
absorbed the same global failure as its own scientific FAIL, and the pass then
reported the bounded envelope exhausted.

### A — engineering / runtime failure

Traceback, schema or type mismatch, wrong path, checkpoint serialization error,
CUDA runtime failure, input that will not resolve.

**Preserve the log and every artifact. STOP. Inspect before re-running.** These
are our bugs; re-running without understanding one risks burning GPU hours on the
same failure or, worse, producing a result under a defect.

### B — search incomplete / interruption

The pass stopped partway; state is preserved and `finalizable: false` is recorded.

**Re-run the identical command with `--resume`.** Nothing else.

### C — scientific envelope exhausted

**Only reachable when candidates really failed on their own merits** — a global
dependency block (A0) aborts before this can be reached, and the two verdicts no
longer share a handler.

If a Track-G or Track-R bounded envelope produces no valid configuration, the run
reports:

```
C7_SOURCE_SEARCH = NEEDS_SCIENTIFIC_DECISION
```

Every attempted configuration is retained in the leaderboard. This is a
**scientific negative result about the source side** and it is preserved as one.

It is **not** permission to widen candidate ranges, run a second pass, change the
optimizer family, switch backbone, add a loss term, change the DET search arm,
tune per arm, or look at any target result. **STOP and report.**

---

## 11. ONNX / C6 warnings

C6 is scientifically closed. Do not change SCRFD or any C6 code in response to
historical ONNX `VerifyOutputSizes` warnings — they did not cause any C6 failure
and C7 is detector training only. If they appear, record them as operational
provenance and carry on.

---

## 12. What to send back

On PASS, collect:

- the final C7 stage block from stdout
- `reports/full/c7/C7_SCIENTIFIC_INPUTS.json`
- `reports/full/c7/C7_SCIENTIFIC_SEARCH_PLAN.json`
- `reports/full/c7/C7_SCIENTIFIC_SOURCE_SEARCH_G.json`
- `reports/full/c7/C7_SCIENTIFIC_SOURCE_SEARCH_R.json`
- `reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_G.json` (summary is enough)
- `reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_R.json` (summary is enough)
- `reports/full/c7/DETECTOR_CONFIG_LOCK.json`
- the `VERIFY_CONFIG_LOCK` check block
- `tracks.G.winner_config_sha256` and `tracks.R.winner_config_sha256`
- per track: `trials_declared`, `trials_executed`, `trials_by_status`
- per track: `logical_occurrences` and `unique_configurations` from the plan's
  `cost` block, plus a count of `C7_TRIAL_OCCURRENCE.json` files (which must
  equal the logical occurrence count, unlike the config-keyed summaries)
- `reports/evidence/quarantine/**/RECOVERY_RECORD.json`, if §9A was run
- `recipe_text_cache` evidence from `C7_SCIENTIFIC_INPUTS.json`: `file_sha256`,
  `cache_identity_sha256`, `rebuilt_at_runtime`
- `search_decision_identity` and `training_arm`
- the C6 DET bank identity actually bound (`search_binding.c6_bank_selected_set_sha256`)
- `source_package_identity`
- `pretrained` (SigLIP2 identity + ConvNeXt weight SHA)
- GPU / runtime provenance (device, driver, torch, AMP dtype)
- `target_access`

Compact log grep:

```bash
grep -nE 'VERIFY_C6_EVIDENCE|SCIENTIFIC_SOURCE_SEARCH|FINALIZE_DETECTOR_CONFIG|VERIFY_CONFIG_LOCK|c7_track_[gr]_|search_plan_identity|search_decision_identity|training_arm|declared_trials|logical_occurrences|unique_configurations|trials_by_status|winner_config_sha256|winner_checkpoint_sha256|NEEDS_SCIENTIFIC_DECISION|SEARCH_STATE_IDENTITY_MISMATCH|GLOBAL_DEPENDENCY_UNAVAILABLE|TextCacheError|recipe_text_cache|BLOCKED|Traceback|target_access' \
  gpu_c7_${HANDOFF_HEAD:0:7}.log
```

And the one-line verdict:

```bash
grep -nE '^C7|scientific_status|engineering_status' gpu_c7_${HANDOFF_HEAD:0:7}.log | tail -20
```

---

## 13. What PASS means

A successful C7 run means **all** of:

- both Track-G and Track-R bounded passes reached `COMPLETED`
- every attempted trial retained, including FAIL and DIVERGED
- exactly one winner per track
- `DETECTOR_CONFIG_LOCK.json` written, with `tracks.G` and `tracks.R`
- strict `VERIFY_CONFIG_LOCK` passed, including re-hashing both winning
  checkpoints on disk
- `target_access = 0`

**It does not mean C8 may be launched.** Exactly one thing stands between C7
PASS and C8: the C7 lock must be **audited** off the returned artifacts, not
merely observed to exist. A file appearing at the expected path is not the same
claim as a verified frozen configuration, and C8 trains 42 rows at whatever that
file names.

### Where detector reliability actually sits

It is a **post-C8 / pre-C9** barrier. It does **not** gate C8.

```
C7    bounded search -> DETECTOR_CONFIG_LOCK
      -> AUDIT the lock                        <- the only gate before C8
C8    the source-only scientific matrix, 42 rows
POST-C8 / PRE-C9
      execute and freeze detector reliability,
      including BA_sep once its remaining protocol decisions are frozen
C9    requires a valid DETECTOR_RELIABILITY_LOCK_C
      before SOURCE_MATRIX_LOCK_C may close
```

`DETECTOR_BA_SEP_PROBE_PROTOCOL`, `DETECTOR_BA_SEP_EVIDENCE_VECTOR` and
`DETECTOR_BA_SEP_PROBE_SEEDS` are still `NEEDS_SCIENTIFIC_DECISION`, and they may
not be chosen from C8 outcomes — picking a probe protocol after seeing which
detectors trained well would make the gate a function of the result it exists to
test. That is a reason **C9** is blocked, not a reason to hold C8: the probe needs
C8's trained detectors to exist before it can run at all.
