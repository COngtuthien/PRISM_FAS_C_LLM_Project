# Applying the M2 → M3A contract hotfix to the Linux RTX 5090 host

Four runtime files change. Nothing else. Your ~34 GB of datasets, weights,
`.venv`, caches **and the partial preprocessing already on that machine** stay
exactly where they are.

    src/prism_fas/pipeline/preparation.py
    src/prism_fas/data/run_context.py
    src/prism_fas/cli/main.py
    train.py

The deployed copy on that machine was taken from commit `787b9b2`.

## What failed, and why

Preparation ran SCRFD over the source corpora for real — the first time this
project has ever reached derived-data preparation — and then stopped:

    [PREPARATION_FAILED]
    derived-data preparation failed at m3a_package:
    FileNotFoundError: <project>/data/processed/manifests/source_frames.parquet

    Stopped BEFORE C4. No scientific work was started.

Two independent path expressions for one artifact:

* the **producer**, `_step_m2`, called the legacy `m2_runner.run`, which writes
  JSONL results and crops under
  `<work_root>/m2/<version>/<config_hash>/m2a/`, and never writes a parquet
  manifest at all;
* the **consumer**, `_step_m3a`, called `build_package(paths.processed_root, …)`,
  whose `load_m2_samples` reads `<input_root>/manifests/source_frames.parquet`.

`data/processed` has never held an M2 manifest under any convention this project
has used. In the inherited Version-B layout it holds built *packages*
(`prism_data_v1_m3b`, `prism_target_eval_v2`, the synthetic banks); Version-C
preparation writes packages to `data/packages`. Nothing has ever written
`data/processed/manifests/`.

The tests could not see it. The preparation suite stubbed `m2_runner.run` with a
fake that created `data/processed/<dataset>/` — the stub wrote where the consumer
read, so producer and consumer agreed only inside the test. `build_package` was
stubbed too, so `load_m2_samples` never ran.

## What changed

**One contract.** Both sides now resolve the location through a single function,
`preparation.m2_output_root(repo)`, which delegates to the project's own
`run_profiles.profile_root` under the `full_preprocessing` profile:

    <work_root>/m2/<preprocessing_version>/<config_hash>/full_preprocessing/
        manifests/   source_frames.parquet, source_crops.parquet,
                     target_frames.parquet, target_crops.parquet,
                     preprocessing_failures.parquet
        crops/       <dataset>/<sample_id>.jpg
        state/       M2_PREPARATION_COMPLETE.json

That directory name is not cosmetic: `build_lock` records
`source_m2_namespace = input_root.name` beside a hard-coded
`source_m2_validation_profile = "full_preprocessing"`, so any other basename
produces an M3A lock that contradicts itself.

**The production producer.** `_step_m2` now drives `PreprocessingRunContext` +
`run_preprocessing` — the same pair `prism data preprocess run --run-profile
full_preprocessing` drives — and not the legacy `m2a` CLI helper. The
`build_preprocessing_run_context` constructor moved from `cli/main.py` to
`data/run_context.py` so the preparation path can reach it without importing the
CLI. That it could not is why it drifted onto the legacy runner in the first
place.

**Completion is measured, not assumed.** The old rule was "`data/processed`
exists and is non-empty", which is equally true of a tree that died halfway
through its first dataset. Completion now means all five canonical manifests
present, no target rows, frame and crop counts equal, every canonical source
record walked, a completion marker whose config hash / detector hash / record
counts still match, and the canonical `validate_full_profile` passing — crop
SHA-256s, decodability, dimensions, orphans, temporaries, target isolation.

**A third defect fell out of running the chain end to end for the first time.**
`build_package` writes `PACKAGE_LOCK.json` with status `building`, and
`_step_m3a` then validated it *strictly*, which can never pass before
`finalize_lock` runs. It now validates loose, finalizes, then validates strict —
the order the canonical CLI uses.

## Your existing partial work

Before applying anything, look at what that machine actually has:

    cd /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project
    /usr/bin/python3 train.py --diagnose-data

It prints, and writes to `reports/preflight/DERIVED_DATA_DIAGNOSIS.json`:

* the M2 config root and which namespaces exist under it;
* the legacy `m2a` root, its crop count and its result files;
* the `full_preprocessing` root, each manifest's row count, the crop count and
  whether the completion marker is there;
* the M2 status, and how many records per dataset are still outstanding;
* the package and pair-plan state;
* what `data/processed` is and is not.

It builds nothing, deletes nothing and reads no target.

**Be prepared for this: the SCRFD work the failed run did is in the `m2a`
namespace and is not reusable.** It holds JSONL results rather than the canonical
parquet manifests, and `migrate_m2a` is contract-locked to the frozen
24/24/12/12/0 acceptance counts, so it cannot carry a full corpus across.
Adopting it would mean inventing a migration for artifacts whose provenance the
contract does not cover, which is exactly what must not happen to scientific
inputs. So the full profile redoes detection into its own namespace.

Nothing deletes the `m2a` tree. It stays where it is, and you can remove it
yourself later once the new tree validates.

From the first `full_preprocessing` run onward, resume is real: completed records
are discovered from the canonical manifests and are not walked again, so an
interrupted run continues rather than restarting.

## Applying it — use Git

    cd /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project
    git fetch origin
    git checkout <NEW_HEAD>          # the authoritative_commit in HOTFIX_MANIFEST.json
    /usr/bin/python3 train.py

That moves exactly the four files and leaves everything else alone. No copying,
no hashes to check by hand, no `.venv` rebuild — this fix changes no dependency
pin.

## Applying it — manual fallback

Only if that host has no route to the remote. Copy the **contents** of this
package over the project root, preserving relative paths:

    PRISM_FAS_C_LLM_Project/
    ├── train.py
    └── src/prism_fas/
        ├── pipeline/preparation.py
        ├── data/run_context.py
        └── cli/main.py

`HOTFIX_MANIFEST.json` and `README_APPLY_HOTFIX.md` are this package's own
documentation — do not copy them into the project root. The per-file SHA-256s,
before and after, are in the manifest; verify with `sha256sum`.

## What must not be touched

    data/raw/       the raw CASIA / MSU / SiW-Mv2 corpora
    data/work/      the M2 work tree, including the partial preprocessing
    weights/        the frozen SCRFD, SigLIP2 and backbone weights
    assets/         recipe banks
    .venv/          the installed cuda-cu129 environment
    runs/  reports/  state/    evidence

No recopy of any of them. No reinstall. No manual deletion — in particular you do
**not** need to delete `data/work` or `data/processed`.

## Then run it

    /usr/bin/python3 train.py

Expect, in order: bootstrap, GPU preflight (including the autograd probe fixed in
the previous hotfix), then derived-data preparation, which now prints per-dataset
progress with an elapsed and remaining estimate rather than sitting silent for
hours. M2 → M3A → M3B → the GPAT pair plan, then C4.

If anything fails it still stops before C4 with a reason code and
`Stopped BEFORE C4. No scientific work was started.` The new codes are
`M2_INCOMPLETE` (the tree did not reach completion; partial work is preserved and
a rerun continues from it) and `TARGET_IN_SOURCE_TREE` (a SiW row appeared in the
source M2 tree — a firewall breach, not something more preprocessing can repair).

## Scientific status

    frozen configs changed      no
    C3 banks changed            no
    SCRFD policy changed        no
    datasets changed            no
    scientific identity         unaffected
    target access               0
    C4-C13 scientific execution NOT_RUN
