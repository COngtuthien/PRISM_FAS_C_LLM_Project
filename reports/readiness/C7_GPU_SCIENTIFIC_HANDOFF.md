# C7 GPU scientific handoff — the first real detector search

**This is an execution handoff, not scientific evidence.** It describes a run
that has not happened. No C7 scientific trial has ever executed, no detector has
been trained, and `reports/full/c7/DETECTOR_CONFIG_LOCK.json` does not exist
anywhere.

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
| **Scientific implementation head** | `32250263ab7b399c9ec995ad06b318fbd357d96d` |
| **Checkout head** | the tip of `portable-one-command-full-run` — call it `HANDOFF_HEAD` |
| Branch | `portable-one-command-full-run` |

`HANDOFF_HEAD` is the commit that added this file, and a file cannot contain its
own hash. Resolve it on the host rather than trusting a transcribed SHA:

```bash
HANDOFF_HEAD=$(git log -1 --format=%H -- reports/readiness/C7_GPU_SCIENTIFIC_HANDOFF.md)
echo "$HANDOFF_HEAD"
```

Everything between `32250263ab7b399c9ec995ad06b318fbd357d96d` and `HANDOFF_HEAD` is documentation
and evidence only — no file under `src/`, `configs/`, `train.py`, `bootstrap.py`
or `assets/` differs. **Verify that before running anything:**

```bash
git diff --stat 32250263ab7b399c9ec995ad06b318fbd357d96d "$HANDOFF_HEAD" -- src train.py bootstrap.py configs assets
```

That must print nothing. If it prints anything, the scientific implementation
moved after `32250263ab7b399c9ec995ad06b318fbd357d96d` and this handoff no longer describes what
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

The identity moved because an implementation defect was corrected at `32250263ab7b399c9ec995ad06b318fbd357d96d`,
**not** because the approved decision changed. `UNIQUE_INHERITED_ANCHOR` was
being executed as "the learning-rate coordinate is not searched", which omitted
Track G's learning rate — the first coordinate of the frozen order — and left
Track G with 12 trials instead of 15. The approved YAML was always correct and
no byte of it changed; only the code that read it did. Zero scientific trials
had run, so nothing needed regenerating. Full account in the supersession record.

If you are comparing against an older handoff or an older search state, this is
the identity that legitimately differs.

---

## 3. The declared cost — read this before launching

```
Track G       15 trials     learning_rate_multiplier, weight_decay, warmup,
                            lambda_syn, lambda_risk
Track R       24 trials     learning_rate_multiplier, weight_decay, warmup,
                            lambda_syn, lambda_local, lambda_MIL, lambda_P,
                            lambda_risk
              --
total         39 trials

per trial     35 epochs x 45 optimizer steps
total         39 x 35 x 45  =  61,425 optimizer steps

effective batch (G5)   12 real-live + 12 real-spoof + 8 synthetic  =  32
```

Every trial runs the **full** frozen M9 schedule. Do not reduce epochs, steps,
batch composition or the number of candidates to fit the machine — L.12 forbids
shrinking a scientific budget to fit hardware, and a shortened schedule would
make the result a measurement of something else.

Track G tunes five coordinates and Track R eight because their active loss sets
genuinely differ: Track G instantiates no ConvNeXt, no regions and no
PromptHead, so `lambda_local`, `lambda_MIL` and `lambda_P` are NOT APPLICABLE
there, and manifold-OFF Track R has no `L_real`/`L_out`/`L_clean` to weight. The
fairness invariant is *one frozen configuration within a track*, not one
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
| 6 | search decision resolves DET / FROZEN | `prism_fas.search.c7_decision.load_decision` |
| 7 | `target_access = 0` | no target path, label or metric is reachable from any C7 code path |

Required inputs, all of which must be present **and valid**:

```
weights/                                 pinned SigLIP2 + ConvNeXt V2 Atto, SHA-verified
data/packages/prism_data_v1_m3b          M3B source package, status validated
reports/full/c6                          the frozen C6 closure
runs/full/c5/scientific/candidates       the rendered candidate bytes the banks address
```

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
— PASS, FAIL or BLOCKED. The C7 lock has to be audited before C8 is considered,
and C8 additionally waits on a detector-reliability barrier that is not yet
frozen.

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
- `search_decision_identity` and `training_arm`
- the C6 DET bank identity actually bound (`search_binding.c6_bank_selected_set_sha256`)
- `source_package_identity`
- `pretrained` (SigLIP2 identity + ConvNeXt weight SHA)
- GPU / runtime provenance (device, driver, torch, AMP dtype)
- `target_access`

Compact log grep:

```bash
grep -nE 'VERIFY_C6_EVIDENCE|SCIENTIFIC_SOURCE_SEARCH|FINALIZE_DETECTOR_CONFIG|VERIFY_CONFIG_LOCK|c7_track_[gr]_|search_plan_identity|search_decision_identity|training_arm|declared_trials|trials_by_status|winner_config_sha256|winner_checkpoint_sha256|NEEDS_SCIENTIFIC_DECISION|SEARCH_STATE_IDENTITY_MISMATCH|BLOCKED|Traceback|target_access' \
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

**It does not mean C8 is ready to run.** Two things stand between C7 PASS and
C8:

1. the C7 lock must be audited off the returned artifacts, not merely observed to
   exist;
2. C9 is blocked on `DETECTOR_RELIABILITY_LOCK_C`, whose probe protocol, evidence
   vector and seed count are still `NEEDS_SCIENTIFIC_DECISION` — and they may not
   be chosen from C8 outcomes.

C8 stays unauthorized until both are addressed.
