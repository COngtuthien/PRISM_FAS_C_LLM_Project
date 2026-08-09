# C0 — Compute backend and portability plan

Version C **MUST NOT** depend on Modal-specific execution semantics. The same scientific
CLI, the same config and the same run identity must execute on either backend. The backend
is **operational metadata**; the scientific treatment factors are identical across it.

C0 **designs** this layer. C0 does not execute a GPU job, a Modal job or an SSH job.

## 1. Why this is urgent

Version B was Modal-centric in practice: every GPU stage ran on Modal L4 (fallback L40S),
and the one PC full-training row was **blocked** for lack of a local CUDA device. Modal
access for this project may expire soon. If the hypothesis-critical C8 matrix cannot start
and finish on Modal, it must run on a lab GPU reached over SSH — and that support has to
exist *before* C8, not be improvised during it.

The inherited architecture already helps: the Version-B trainer core never imports
`modal`; the Modal app is a wrapper that mounts volumes, selects a GPU and calls the same
CLI. Version C keeps that rule and adds a second wrapper.

## 2. Official backends

| Backend id | Role | Precision | Scientific status | Switch rule |
|---|---|---|---|---|
| `modal_l4` | preflight / temporary engineering training while access exists | BF16 AMP | valid if a **whole declared block** completes on it | never assume future availability |
| `ssh_lab_bf16` | preferred final matrix when the lab GPU supports BF16 | BF16 AMP | preferred homogeneous C8/P3 profile | capability + portability gate first |
| `ssh_lab_fp16` | fallback only if the lab GPU lacks BF16 | FP16 AMP | a **new** compute profile; cannot be pooled with BF16 | freeze before C8, or restart the affected block |
| `cpu_fixture` | unit and smoke tests | FP32 | engineering only | never used for a scientific metric |

`modal_l4` and `ssh_lab` are the two backend ids; `ssh_lab_bf16` / `ssh_lab_fp16` are the
two precision profiles of `ssh_lab`.

## 3. `modal_l4` — the temporary backend

- May be used immediately for C0–C7 preflight, GPAT-C engineering, and any complete
  scientific block that can finish before access expires.
- A block that is interrupted mid-way by expiry is **not** partially valid: the affected
  rows are rerun on the final profile.
- Volumes hold the frozen processed source package and the synthetic banks by hash. No
  backend-specific preprocessing branch is permitted.

## 4. `ssh_lab` — the future lab backend

**Mandatory before C8.** Requirements:

- Runs survive terminal disconnection: `tmux`, `screen`, `systemd-run` or `nohup`,
  whichever the lab host provides.
- SSH private keys and the Gemini API key are environment secrets. Never committed, never
  logged, never placed in provenance JSON.
- Checkpoint at least every epoch **and** at a fixed step interval. Atomic writes
  (write temp → fsync → rename), then `rsync`/`scp` with checksums and partial-transfer
  support.
- A completed remote run is valid only after **local** verification of the checkpoint,
  config and metrics hashes.
- Raw datasets stay local or on approved lab storage. The remote consumes the same frozen
  processed package and synthetic-bank artifacts **by hash**.

### 4.1 The lab GPU is unknown, and stays unknown until probed

**Do not assume a GPU model.** Before C8, a capability probe must record and freeze:

```
GPU name
VRAM
driver version
CUDA runtime
PyTorch build (torch + CUDA build string)
BF16 support        (bool)
FP16 support        (bool)
hostname class
```

Decision rule:

- **BF16 available** → use the frozen BF16 scientific profile (`ssh_lab_bf16`). Preferred,
  because it matches the Modal profile and keeps the matrix homogeneous.
- **BF16 unavailable** → **STOP** and freeze an explicit FP16 lab profile
  (`ssh_lab_fp16`) *before* any hypothesis-critical training. Do not pool BF16 and FP16
  seeds as if identical without a preregistered backend factor.

> **Never silently change precision inside a scientific block.**

## 5. Run-identity capture

Every run records, regardless of backend:

```
backend_id            modal_l4 | ssh_lab
gpu_model, vram, driver, cuda_runtime, torch_cuda_build
hostname_class
precision_mode        bf16_amp | fp16_amp | fp32
microbatch
gradient_accumulation_steps
effective_batch_size
seed, data order identity, optimizer schedule, total step count
```

If the backend changes between runs, the **effective batch size, data order, seeds,
optimizer schedule and step count remain identical**. The microbatch may change only via
gradient accumulation, so the effective composition over the accumulation window is
preserved. Never mix GPUs or precisions inside a single run.

## 6. Portability test contract (Modal → lab)

Bounded checks, to be run before the transition. Each is engineering evidence.

| # | Check | Pass criterion |
|---|---|---|
| 1 | same frozen fixture | both backends load the identical fixture by hash |
| 2 | same checkpoint load | a checkpoint written on one backend loads on the other with no missing/unexpected keys |
| 3 | forward comparison | logits on the frozen fixture agree within a declared numerical tolerance |
| 4 | finite loss | loss finite on the fixture batch |
| 5 | one optimizer step | a single step completes and changes parameters |
| 6 | checkpoint save | atomic write, hash recorded |
| 7 | checkpoint reload | reload reproduces model, optimizer, scheduler, scaler and RNG state |
| 8 | resume | resumes at the correct batch/epoch boundary |
| 9 | sample ordering | identical sample id sequence for the same seed and epoch |
| 10 | effective batch size | identical, including under gradient accumulation |
| 11 | precision metadata | recorded and matching the declared profile |

> This proves **engineering portability only**. It does **not** establish complete
> training-trajectory equivalence between different GPUs, and Version C must not claim
> that it does.

## 7. Homogeneity rule for the hypothesis-critical matrix

Hypothesis-critical rows **SHOULD** run on one homogeneous final compute profile.

Preferred plan: use Modal for engineering through C7, then run the complete C8 / P3
hypothesis-critical matrix on the lab GPU if Modal availability cannot cover the whole
matrix. If a backend switch happens mid-matrix, the affected rows are rerun rather than
pooled, or the backend is registered as a preregistered factor — never assumed harmless.

## 8. What C0 delivers, and what it does not

Delivered by C0: this contract, the backend id vocabulary, the profile table, the required
probe fields, the identity-capture list and the portability checklist — all recorded in
[`configs/version_c/c0_frozen_design.yaml`](../../configs/version_c/c0_frozen_design.yaml)
under `compute:`.

Not delivered by C0, by design: any GPU execution, any Modal job, any SSH connection, any
capability probe result, any lab GPU model. The lab GPU capability profile is an **open
item** and is listed as such in [`C0_MILESTONE_EXECUTION_PLAN.md`](C0_MILESTONE_EXECUTION_PLAN.md).
