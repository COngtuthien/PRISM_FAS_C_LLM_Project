# C9_DETECTOR_BA_SEP_OPTION1_V2 — pre-execution group-split correction

**NO BA_SEP METRIC WAS EVER OBSERVED — under V1, or before this V2 freeze.**
Nothing in the V1 freeze task, this task, or any run in between computed,
loaded or inspected a BA_sep number, a residual-sensitivity number, or any
other reliability metric. This correction is not a response to a result; it
is a response to re-reading V1's own group-safety claim against its actual
implementation and finding they did not match.

This document freezes V2 as the superseding, corrected protocol for
`synthetic_vs_real_spoof_probe`. **It does not run it.** `--execute` on the
new scientific runner still refuses on every host today, including this one.

---

## 1. The defect this corrects

V1 (`reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.md`, §8) declared:

> Group-safe (the same stable identity always lands in the same bucket for a
> given seed/domain) ... (`sample_id` for real records, `synthetic_id` for
> synthetic records; both are the canonical stable identities their
> respective readers already expose).

That declaration conflated two different things under one name,
`stable_group_identity`:

- **Group identity** — which underlying source video/record a sample was
  ultimately derived from. This is what a group-safe split must partition
  on, so that no two samples sharing an origin can land on opposite sides of
  train/validation.
- **Sample identity** — the individual crop/candidate's own identity. Fine
  for deterministic *selection*, wrong for a *split*.

V1 used `sample_id` (real) / `synthetic_id` (synthetic) — SAMPLE identities —
as the split GROUP key. Multiple canonical crops can share one
`source_record_id`, and multiple synthetic candidates generated from the
same live source sample (multiple recipes, multiple routes) share one
`live_target_sample_id` and hence one underlying `source_record_id`, while
each carries its own distinct `synthetic_id`. Under V1's rule, two samples
from the same source record — including a REAL spoof sample and a SYNTHETIC
candidate generated from it — could be assigned to different buckets by
uncorrelated chance, because their sample identities differ even though
their group identity is the same. V1's `group_safe: true` was therefore an
unverified claim, not a proven property.

## 2. The correction

`PopulationRecord` (`src/prism_fas/evaluation/synthetic_real_probe.py`) now
carries two distinct fields instead of one:

```
sample_identity: str        # sample_id (real) / synthetic_id (synthetic)
stable_group_identity: str  # source_record_id, for BOTH populations
label: int
source_domain: str
```

- **Real population** (`resolve_real_spoof_population`): `stable_group_identity`
  is read directly off the `source_train` manifest row's `source_record_id`
  field — the same field `prism_fas.detector.dataset.M9ValidationDataset.record_of`
  already exposes. Fail closed (raise `SyntheticRealProbeError`) if a row's
  `source_record_id` is empty.
- **Synthetic population** (`resolve_synthetic_population`): `stable_group_identity`
  is resolved as `live_target_sample_id -> source_train row -> source_record_id`,
  through the exact same `sample_id -> source_record_id` lookup the real
  population uses (`_source_record_id_by_sample_id`, one implementation,
  never a second). Fail closed if `live_target_sample_id` is empty, or does
  not map to exactly one `source_train` row.
- **The split** (`split_bucket`/`assign_splits`) now partitions ONLY on
  `stable_group_identity`. **The selection order**
  (`_selection_order_key`/`balance_classes`) now orders ONLY on
  `sample_identity`. These are proven, by regression, to be genuinely
  different axes — not the same key reused under two names.
- **`verify_group_safe_split`** (new): asserts, for any split, that train and
  validation share zero `stable_group_identity` values. Called by the
  scientific runner before writing a binding artifact, and exercised
  directly against both a safe and a deliberately leaking fixture split.
  This is defense-in-depth on top of `split_bucket`'s already-safe
  construction — proving the property again, rather than trusting the V1
  declaration a second time.

## 3. What did NOT change

Per the freeze task's explicit retention list, byte-identical to V1:

- The evidence vector: `[global_logit_G, p_global]`, 2-D, same forbidden
  fields.
- The checkpoint policy: 5 checkpoints/arm, 15 total, same resolver
  (`resolve_checkpoint_set`), same arithmetic-mean aggregation.
- The probe seeds: `[20260806, 20260807, 20260808]`, same non-inheritance
  disclosure.
- The classifier: `torch.nn.Linear(2, 1)`, zero-init, `BCEWithLogitsLoss`,
  L2 `lambda=1e-4` on weight only.
- The solver: full-batch `torch.optim.LBFGS` with the identical frozen
  hyperparameters (`lr=1.0, max_iter=200, max_eval=250,
  tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
  line_search_fn="strong_wolfe"`), CPU, `float64`.
- The threshold: `0.5`.
- The metric: `prism_fas.train.metrics.balanced_accuracy`, per-seed then
  arithmetic mean over exactly three seeds.
- The ceiling: `BA_sep <= 0.75`.
- The target firewall: `target_access: 0`.

## 4. New in V2

- **Sample/group identity separation** (§2 above).
- **The 80/20 split hash namespace changes** from `c9-ba-sep-option1-v1` to
  `c9-ba-sep-option1-v2` — a result-affecting change (bucket assignment can
  differ from V1's), so it must move the protocol identity, and it does
  (§6).
- **A new, separate sample-selection-order hash rule**, explicitly keyed on
  `sample_identity` — was previously (in V1) the same identity the split
  used; now genuinely independent.
- **`balance_report`** (new function): `balance_classes` plus
  `unique_source_record_id_counts` (pre- and post-balance, per population) —
  so a reviewer can see how much group diversity balancing preserved
  without recomputing it by hand.
- **An explicit all-arm hard verdict rule**, frozen before any metric exists
  (`synthetic_real_probe.hard_verdict`, `configs/evaluation/c9_detector_ba_sep_option1_v2.yaml::hard_verdict_rule`):

  ```
  PASS iff BA_sep_RND <= 0.75 AND BA_sep_DET <= 0.75 AND BA_sep_LLM <= 0.75
  ```

  A single arm above the ceiling fails the whole test; there is no
  partial-arm pass. Independent of `C_H4_SUPPORT_RULE`, unchanged.
- **A scientific runner CLI**
  (`src/prism_fas/evaluation/synthetic_real_probe_runner.py`,
  `python -m prism_fas.evaluation.synthetic_real_probe_runner`), described
  in §7.

## 5. Protocol config and identity

`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml` — `status:
FROZEN_NOT_RUN`, `decision_id: C9_DETECTOR_BA_SEP_OPTION1_V2`, `supersedes:
configs/evaluation/c9_detector_ba_sep_option1_v1.yaml`,
`supersession_reason: PRE_EXECUTION_GROUP_IDENTITY_CORRECTION`,
`superseded_protocol_identity` bound to V1's frozen identity
(`a6da0ce75ebd92589ea61cba24a85bf8d8144bdbb99f7ec54d31066a66594908`),
`no_ba_sep_observed_under_v1: true`, `no_ba_sep_observed_before_v2_freeze:
true`.

`prism_fas.evaluation.detector_reliability.PROBE_PROTOCOL_CONFIG_PATH` now
points at V2. V1's config file (`c9_detector_ba_sep_option1_v1.yaml`) is
**untouched** — its own identity, re-verified by regression, still resolves
to the value frozen for it. `barrier.protocol_identity` applied to the V2
payload produces a NEW 64-hex-character identity, distinct from V1's, proven
by regression to differ.

## 6. Checkpoint decision-logit contract

`detector_checkpoint_identity.required_decision_logit_name: global_logit_G`
is now explicitly declared in the frozen config and explicitly checked by
the scientific runner's `--bind-only` mode before any binding is written —
a protocol that claimed a different decision logit name would be refused,
not silently bound.

## 7. The scientific runner

`src/prism_fas/evaluation/synthetic_real_probe_runner.py`
(`python -m prism_fas.evaluation.synthetic_real_probe_runner`), three
mutually-exclusive modes:

- **`--preflight-only`** — read-only, all arms. Validates the frozen
  protocol, reports its identity, and best-effort reports whether the 15
  real C8 checkpoints are resolvable on this host (never fatal if not).
  Never fits a probe, never writes a file.
- **`--bind-only --arm {RND,DET,LLM}`** — resolves the real checkpoint
  manifests and populations for one arm through the exact canonical readers
  `synthetic_real_probe.py` already wraps, builds the group-safe split and
  balanced selection for every `(probe_seed, source_domain)` cell, verifies
  group safety, and writes ONE binding-record artifact
  (`reports/full/c9/ba_sep_option1_v2/BINDING_{ARM}.json`) — but never loads
  a checkpoint's weights and never opens an image. Writes nothing on any
  failure.
- **`--execute --arm {RND,DET,LLM}`** — would run the real probe end to end.
  Calls `synthetic_real_probe.run_scientific_probe`, which still raises
  `NotImplementedError` on every host today. This CLI reports that refusal
  clearly and exits non-zero; it never falls through to a partial or
  approximate computation.

**On this development laptop**, which has no `runs/full/c8/`,
`--preflight-only` exits PASS (the protocol resolves; per-arm checkpoint
resolution is honestly reported as unresolved), `--bind-only` exits BLOCKED
for every arm and writes nothing, and `--execute` exits BLOCKED for every
arm and writes nothing. All three were run on this host as part of this
task; none produced a BA_sep value, loaded a checkpoint, opened an image, or
wrote a scientific artifact.

## 8. Tests

`tests/pipeline/test_c9_ba_sep_option1_protocol.py` — the 44 V1/shared
mechanics tests, `PopulationRecord` fixtures updated to the two-field
identity shape (no test behavior changed; the split/balance/normalization/
probe/aggregation mechanics are proven exactly as before).

`tests/pipeline/test_c9_ba_sep_option1_v2_runner.py` — 63 new tests: V1
untouched; V2 supersession metadata and identity; group-identity resolution
for both populations with fail-closed coverage (empty/ambiguous
`source_record_id`, empty/unmapped `live_target_sample_id`); the explicit
proof that a real sample and a synthetic candidate sharing one source record
resolve to the same group and the same split bucket; that split and
selection are genuinely different axes; `verify_group_safe_split` accepting
a safe split and rejecting a leaking one; `balance_report`'s group-count
fields; the hard verdict rule (pass/fail/boundary/missing-arm); and the
runner CLI's three-mode contract (usage errors, fail-closed refusal on this
host, deterministic binding-artifact content, group-safety verified before
write, no checkpoint weights or images ever touched).

123 tests pass across the three C9-scoped files
(`test_c9_ba_sep_option1_protocol.py`,
`test_c9_ba_sep_option1_v2_runner.py`, `test_c9_scientific_evidence.py`).

## 9. A latent bug found and fixed while making this correction

`resolve_real_spoof_population`/`resolve_synthetic_population` (as
originally written in the V1 freeze) constructed their source-package reader
with a bare `LoaderConfig()` — pydantic requires every field
(`loader_schema_version`, `package`, `image`, `label_mapping`, `sampler`,
`backends`, `dataloader`) and raises `ValidationError` on an empty
constructor call. This was unreachable in the V1 test suite (no V1 test
calls either resolver against a real repo) so it was never observed to
fail, but it would have crashed on first real use on the GPU host. Both
resolvers now build their reader through
`prism_fas.data.loader.config.load_loader_config(repo / "configs/data/loader_m4.yaml")`
— the exact canonical loader config C8/`M9Trainer` itself uses
(`src/prism_fas/pipeline/adapters/c8.py:1361`) — never a second
configuration invented for this module.

## 10. What this correction does NOT do

- Does not compute a real BA_sep value.
- Does not resolve any of the nine `REQUIRED_DETECTOR_RELIABILITY_TESTS`.
- Does not touch `crop_padding_interpolation`'s data-blocked status.
- Does not create `DETECTOR_RELIABILITY_LOCK_C.json`.
- Does not run C9, C10, C11, C12 or C13.
- Does not access target data. `target_access = 0` throughout, on every
  path, including the runner CLI.
- Does not retrain C7 or any C8 row, add `C-R-RND`, or alter any C7 winner.
- Does not wire `run_scientific_probe`. It remains a deliberate
  `NotImplementedError`, exactly as V1 left it.

**C9 remains correctly BLOCKED after this task.** Correcting one test's
protocol pre-execution mechanics, out of nine required tests, neither
computes nor fabricates a pass for `detector_reliability.verify_lock()`.
