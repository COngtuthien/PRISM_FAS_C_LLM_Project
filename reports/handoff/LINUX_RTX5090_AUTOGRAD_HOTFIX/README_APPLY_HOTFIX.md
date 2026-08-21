# Applying the autograd preflight hotfix to the Linux RTX 5090 host

Two runtime files change. Nothing else. Your ~34 GB of datasets, weights,
`.venv` and caches stay exactly where they are.

    src/prism_fas/pipeline/gpu_preflight.py
    src/prism_fas/evaluation/variant_audit.py

The deployed copy on that machine was taken from commit `1f2e24f`.

## What failed, and why

The run got all the way through:

    Environment       cuda-cu129
    Device            CUDA
    Bundle            PASS 24/24
    Disk              PASS
    Write access      PASS
    Target firewall   ARMED
    Execution intent  GPU_SCIENTIFIC_FULL
    Stage range       C4 -> C13

and then stopped:

    [AUTOGRAD_FAILED]
    the representative model could not complete a forward/backward step:
    AttributeError: 'Tensor' object has no attribute 'image'

That is the preflight refusing to start C4, which is exactly what it is for —
but it was refusing for a defect in itself, not in your machine. `PRISMDetector.
forward` takes a `DetectorBatch`: an image, its nine region priors, visibility,
labels and the synthetic provenance the declared losses read. The probe was
calling it with a bare `torch.randn(2, 3, 224, 224)`. It had invented an input
contract the trainer does not use, so it could only ever have passed by accident.

Your GPU, driver, torch build and CUDA profile were never the problem. The three
gates before this one — kernel launch, arch list, VRAM — all passed on the 5090.

## What the probe does now

It builds its batch through the same contract the trainer uses:
`batch_contract_for("G5", ...)` for the composition, `audit_batch(...)` for the
tensors, then `.to(device)`. It runs the real loss graph through
`compute_losses`, and before letting C4 start it proves, on the selected CUDA
device:

* the forward completes;
* the loss is a scalar and finite;
* the backward completes;
* **every** trainable parameter receives a gradient — not "some parameter
  somewhere", which passes on a graph that has silently detached a branch;
* every gradient is finite;
* the batch, the output, the loss, the parameters and the gradients are all on
  the CUDA device that was selected — no silent CPU fallback.

It opens no dataset, resolves no target, writes no artifact, and takes about two
seconds on a CPU, so rather less on a 5090.

A second defect was fixed behind the first one. The audit stub tower drew its
tokens on the host and returned them unmoved; once the input contract was right,
that would have been the next failure, inside `region_embeddings`, mixing host
and device tensors. It now follows the input image's device. CPU values are
bit-identical, so no CPU audit or rehearsal result changes.

## Applying it — use Git

    cd /path/to/PRISM_FAS_C_LLM_Project
    git fetch origin
    git checkout <NEW_HEAD>          # the authoritative_commit in HOTFIX_MANIFEST.json

That moves exactly the two files and leaves everything else alone. It is the
preferred method: no copying, no hashes to check by hand.

## Applying it — manual fallback

Only if that host has no route to the remote. Copy the **contents** of this
package over the project root, preserving relative paths:

    PRISM_FAS_C_LLM_Project/
    └── src/prism_fas/
        ├── pipeline/gpu_preflight.py
        └── evaluation/variant_audit.py

`HOTFIX_MANIFEST.json` and `README_APPLY_HOTFIX.md` are this package's own
documentation — do not copy them into the project root.

Verify afterwards:

    sha256sum src/prism_fas/pipeline/gpu_preflight.py
    # 39f876bafe131513ba6afb1dba3cf66bf470649b292e83170a3a513d4ad6ca6e

    sha256sum src/prism_fas/evaluation/variant_audit.py
    # 69c285294e5a07f45db71aa953c046f7908d4959f09ba0b6cb4f1b7f44579fb4

Before the copy, the deployed files should read:

    14be3b1c06a385935aec6740e9fef9a12c2d743b4f65743eff96d4bc7cbfa655   gpu_preflight.py
    6b21a16c1e494881cec63cea2562b458c60b250f5f878d91d9c69eb8cb187577   variant_audit.py

Files ship with LF endings, which is what a Linux checkout writes, so a manual
copy lands byte-identical to what Git would have written.

## What must not be touched

    data/       the raw CASIA / MSU / SiW-Mv2 corpora
    weights/    the frozen SCRFD, SigLIP2 and backbone weights
    assets/     recipe banks
    .venv/      the installed cuda-cu129 environment
    runs/  reports/  state/    evidence

No recopy of any of them. No reinstall. No `.venv` rebuild — the torch build on
that host is the right one and this fix does not touch a single dependency pin.

## Then run it

    python train.py

Same command as before. The preflight will run again first. If it passes you
will see `GPU preflight PASS` with the probe count and the device line, and the
run continues into C4.

If it fails again it will still stop with a reason code and
`Stopped BEFORE C4. No scientific work was started.` — that behaviour was not
weakened, and the probe was not disabled to make the run proceed.

## Scientific status

    frozen configs changed      no
    C3 banks changed            no
    scientific identity         unaffected
    target access               0
    C4-C13 scientific execution NOT_RUN
