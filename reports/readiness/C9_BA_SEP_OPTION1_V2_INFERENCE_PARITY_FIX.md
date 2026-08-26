# C9_DETECTOR_BA_SEP_OPTION1_V2 — detector-inference parity fix

**DISCOVERED AFTER SUCCESSFUL BINDING, BEFORE THE FIRST REAL BA_sep.** The
real GPU host has already run `--preflight-only` and `--bind-only`
successfully and stably:

- protocol identity: `720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`
- checkpoint binding identity: `fa380fa8e732f8536fe175d449542e636563d92d8d75f64bb07b40ca180f63b0`
- population plan identity: `90d00d9f4bb50a93724d1ac6a632d6fa5052cf2d7ec0d08989c4c7004fa6cae1`
- 15/15 checkpoints bound, 12 population cells bound.

**No checkpoint had yet been loaded for the real probe. No detector forward
had yet occurred for the real probe. No BA_sep had yet been observed.** A
final implementation audit, run before authorizing the first real
`--execute`, found that detector inference did not yet match the canonical
C8 execution path in three respects. This document records the fix. **It
does not change the V2 scientific protocol.**

---

## 1. The three parity gaps

### 1.1 Hard-coded CPU device

`construct_row_trainer` passed `device="cpu"` to `M9Trainer` unconditionally.
C8's own scientific row executor
(`pipeline.adapters.c8._run_scientific_row`) resolves its device through
`pipeline.adapters.c7._scientific_device()` — CUDA, or a fail-closed refusal;
never a silent CPU fallback, because a scientific detector forward under a
different precision contract (`amp`, bf16/fp16) than the one the checkpoint
was actually written under would not be evaluating the same detector. The
runner's evidence-forwarding path had never been exercised against a real
CUDA host, so this mismatch was never observed as a failure — it would have
either silently run in the wrong precision contract or crashed loading a
CUDA-trained checkpoint's tensors onto a CPU-only construction, depending on
how PyTorch happened to handle it.

### 1.2 A CUDA tensor cannot reach `np.asarray` directly

`extract_evidence` called `np.asarray(model_output.global_logit)` directly.
A CUDA tensor raises (`TypeError: can't convert cuda:0 device type tensor to
numpy. Use Tensor.cpu() to copy the tensor to host memory first.`) under a
bare `np.asarray`; C8's own canonical evaluation path
(`pipeline.adapters.c8._cross_source_evaluation`) always converts with
`.detach().float().cpu().numpy()` first. Because every test and every prior
run of this module had only ever run on this CPU-only development laptop,
this was never observed as a failure either.

### 1.3 One-sample-at-a-time forwarding

`forward_evidence_for_records` built and forwarded a batch of exactly one
sample per call. Functionally survivable, but not the canonical C8
evaluation shape (`M9Trainer`'s own cross-source evaluation batches by
`trainer.config.validation_batch_size`) — and, on a real GPU host with real
population-plan cardinalities (several hundred samples per arm), one-forward-
per-sample would also have been needlessly slow relative to the row's own
frozen batch size.

None of these three gaps is a scientific decision. All three are
implementation parity with the C8 execution path the frozen V2 protocol
already assumes.

## 2. The fix

### 2.1 Exact C8 scientific device resolver

`construct_row_trainer` now resolves `device = _scientific_device()`
(imported from `prism_fas.pipeline.adapters.c7`, the same function C8's row
executor imports and calls) BEFORE anything else — before the C7 lock is
even read, before the C6 bank is opened. If the scientific device cannot be
resolved (no CUDA on this host), `construct_row_trainer` fails closed
immediately: no lock verification, no bank open, no trainer construction
occurs. The resolved `device` is passed to `M9Trainer(..., device=device,
...)` — never a hard-coded string.

### 2.2 CUDA-safe evidence conversion

A new `_evidence_scalar(value)` helper converts any `torch.Tensor` with
`.detach().float().cpu().numpy()` before reducing it to a Python float;
non-tensor values (plain floats, numpy arrays — what every existing fixture
test already passes) pass through unchanged. `extract_evidence` now calls
`_evidence_scalar` for both `global_logit` and `p_global`. Proven on this
CPU-only laptop with a `requires_grad=True` tensor, which fails a bare
`np.asarray` for the same *category* of reason a CUDA tensor does (a
conversion guard, not device-transparent) — `_evidence_scalar` handles it
correctly where a bare `np.asarray` would raise.

The linear probe itself remains exactly as frozen: CPU, `float64`,
unchanged. Only the DETECTOR forward's evidence extraction is now
device-transparent; the probe fit that consumes the resulting `[float,
float]` pairs never sees a tensor of any kind.

### 2.3 Batched forward evaluation

`forward_evidence_for_records` now builds every required item once, chunks
them by `trainer.config.validation_batch_size` (the row's own frozen
config, never an independent constant), and for each chunk: `collate_items`
(unchanged, the exact canonical batch assembly), `.to(trainer.device)`, one
`trainer.model(batch)` call under `torch.no_grad()` (model eval mode is set
once, by `construct_row_trainer`, before this function is ever called), then
slices each sample's `[global_logit, p_global]` out of the batched
`ModelOutput` and hands that one-sample slice to the unchanged, frozen
`extract_evidence` — never a second 2-field extraction rule, and never a
sample reordered, dropped or merged with another across a chunk boundary.

### 2.4 Explicit pre-forward package/bank reverification

`execute_joint_probe` now calls `sources.verify_detector_inputs(repo,
arms=ARMS)` once, immediately after its identity/self-hash checks and
BEFORE the checkpoint-count guard or any checkpoint construction, and
requires:

- the CURRENT source package identity equals
  `checkpoint_binding["source_package_identity"]`;
- for EACH of RND, DET and LLM, the CURRENT C6 arm bank identity equals
  `checkpoint_binding["c6_bank_identities"][arm]`.

Any disagreement raises immediately — zero model constructions, zero
forwards, zero BA_sep, and the specific mismatched arm (or the package) is
named in the refusal. This was previously only ever going to surface as an
eventual `RunIdentity` mismatch deep inside the FIRST checkpoint's strict
load (`construct_row_trainer` → `checkpoint.load_checkpoint`) — still a
correct refusal, but not a preregistered, named, up-front precondition. It
now is.

## 3. What did NOT change

- Every existing guard: protocol identity, checkpoint binding self-hash,
  population plan self-hash, exactly 5 checkpoints/arm and 15 total,
  checkpoint byte-SHA re-verification (still re-checked a second time
  inside `construct_row_trainer`, on top of the new up-front package/bank
  check), the checkpoint's expected `RunIdentity`, C7 lock verification,
  `decision_logit_name == global_logit_G`, `model.eval()`, `torch.no_grad()`,
  `target_access = 0` — all unchanged, all still exercised.
- The V2 scientific protocol
  (`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml`) — untouched;
  identity `720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`
  confirmed unchanged by direct check. No V3 was created.
- The evidence vector (`[global_logit_G, p_global]`), the 5-checkpoint
  arithmetic-mean aggregation, the three probe seeds, the group-safe
  population plan, the balancing rule, z-score normalization, the LBFGS
  linear probe (CPU, `float64`), the 0.5 threshold, the 0.75 ceiling and the
  all-arm hard verdict — all byte-for-byte unchanged.
- `build_checkpoint_binding` and `build_population_plan`'s returned
  structure — untouched. Their identities are what the GPU host already
  bound (§ above); nothing in this fix could have moved them, since neither
  function was edited. A fresh `--preflight-only` / `--bind-only` on the GPU
  host after this fix is expected to reproduce the SAME two identities and
  report `reused: true`.

## 4. Tests

`tests/pipeline/test_c9_ba_sep_option1_v2_runner.py`, new section
**H6. inference parity with C8**: `construct_row_trainer` never hard-codes
`device="cpu"`; it references and calls `c7._scientific_device()`, and
passes the resolved `device` into `M9Trainer`; a device-resolution failure
blocks before the C7 lock or C6 bank are ever touched (proven with a spy
that fails the test if reached); `_evidence_scalar`'s source contains the
`.detach()/.float()/.cpu()` chain, and is proven, on this CPU-only laptop,
to correctly convert a `requires_grad=True` tensor that a bare `np.asarray`
cannot; `extract_evidence` reads `global_logit`/`p_global` correctly and
the evidence dimension remains exactly 2; `forward_evidence_for_records`
batches by `trainer.config.validation_batch_size` (proven with a spy model
against a batch size smaller than the sample count — 5 samples at
batch_size=2 yields exactly 3 forward calls of sizes 2/2/1), runs under
`torch.no_grad()` (proven by checking `torch.is_grad_enabled()` inside the
spy model's own forward call), gives every selected sample exactly one
evidence vector, and does not reorder or mix samples across chunk
boundaries (proven with per-sample-id-encoded fixture evidence);
`execute_joint_probe` blocks on a source-package identity mismatch and on
each of the three arms' C6 bank identity mismatches individually, in every
case BEFORE `construct_row_trainer` is ever called (proven with a spy that
fails the test if reached), while still enforcing the exact-15-checkpoint
guard and the on-disk checkpoint-SHA guard; the V2 protocol identity is
confirmed unchanged.

173 tests pass across the three C9-scoped files
(`test_c9_ba_sep_option1_protocol.py` 44, `test_c9_ba_sep_option1_v2_runner.py`
113, `test_c9_scientific_evidence.py` 16). Broad regression
(`tests/c7 tests/pipeline`): 33 failed / 1686 passed / 22 skipped — the same
pre-existing baseline, unchanged.

## 5. What this fix does NOT do

- Does not compute a real BA_sep value. `--preflight-only`, `--bind-only`
  and `--execute` were each run once against the real repo on this
  development laptop as part of this task and each correctly exited
  `BLOCKED` (no CUDA, no M3B package, no `runs/full/c8/`), writing no file.
- Does not touch the checkpoint binding or population plan already frozen
  on the GPU host — neither `build_checkpoint_binding` nor
  `build_population_plan` was edited.
- Does not resolve any of the nine `REQUIRED_DETECTOR_RELIABILITY_TESTS`.
- Does not create `DETECTOR_RELIABILITY_LOCK_C.json`.
- Does not run C9, C10, C11, C12 or C13.
- Does not access target data. `target_access = 0` on every path.
- Does not retrain C7 or any C8 row, add `C-R-RND`, or alter any C7 winner.
- Does not change the V2 scientific protocol or create a V3.

**C9 remains correctly `BLOCKED_PENDING_DETECTOR_RELIABILITY_SCIENTIFIC_DECISION`
after this task.** Fixing detector-inference parity with C8 neither computes
nor fabricates a pass for `detector_reliability.verify_lock()`.
