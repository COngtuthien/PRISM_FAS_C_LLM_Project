# C9_DETECTOR_BA_SEP_OPTION1_V1 — protocol freeze

**USER APPROVAL RECEIVED BEFORE METRIC OBSERVATION.** The user explicitly
approved Option 1 (common Track-G decision evidence) from
`reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md`, before this
session, before this repository, and before anyone had observed any BA_sep
value or any other detector-reliability metric.

**NO BA_SEP METRIC WAS OBSERVED BEFORE PROTOCOL FREEZE.** Nothing in this
task computed, loaded, or inspected a BA_sep number, a residual-sensitivity
number, or any other reliability metric. Every numeric mechanism this task
implemented was proven correct against clearly-synthetic fixture arrays, not
real detector evidence.

This document freezes and implements the executable protocol for
`synthetic_vs_real_spoof_probe`. **It does not run it.**

Audited/implemented at source HEAD: see the git section of this session's
final report.

---

## 1. The decision

**Decision ID:** `C9_DETECTOR_BA_SEP_OPTION1_V1`
**Decision:** Use the common Track-G detector decision representation for the
synthetic-vs-real BA_sep probe.
**Reason:** Track G is the only frozen Version-C primary detector
representation that exists for all three RND/DET/LLM arms under the same
architecture — confirmed in the dossier's §C by reading
`src/prism_fas/detector/contracts.py` and `src/prism_fas/pipeline/adapters/c7.py`
directly, not assumed.

## 2. Evidence vector

```
[global_logit_G, p_global]
```

Exactly these two fields. **Not added:** `s_region`, `region_distances`,
`local_logits`, `region_embeddings`, SigLIP `z_global`, generator identity,
route identity, quality `q`, or recipe metadata.

**Reduction disclosure**, verbatim from the frozen config
(`configs/evaluation/c9_detector_ba_sep_option1_v1.yaml::evidence_vector_definition.reduction_disclosure`):

> a REDUCED 2-D probe relative to the historical Version-B regional 11-D
> vector [p_global, s_region, nine normalized regional distances]. Track G
> structurally cannot emit the regional terms (no manifold, no region fusion
> — spec §13.1 NORMATIVE), and no Version-C primary row for any arm emits
> them either (primary Track R trains with manifold OFF). This vector MUST
> NOT be described as reproducing the historical 11-D vector.

## 3. Checkpoint binding

**Policy:** all five C8 P3-ready Track-G training-seed checkpoints per arm
(seeds `20260806, 20260807, 20260808, 20260809, 20260810`) — **15 total**,
resolved from real C8 run manifests via
`prism_fas.evaluation.source_evidence.load_row_evidence` (the exact reader C9
itself already uses), re-hashed against the checkpoint bytes on disk. No
checkpoint is chosen by best/median/lowest-ACER or by examining evidence. No
P1/P2 checkpoint participates. `prism_fas.evaluation.synthetic_real_probe.resolve_checkpoint_set`
raises rather than returning a partial set if any of the five is missing,
non-PASS, or fails hash verification.

`identities: {}` in the frozen config — the real 15 SHA-256 values are bound
only when this runs on a host that possesses `runs/full/c8/`. This
development clone does not, and none was invented here.

## 4. Checkpoint aggregation

For arm `A` and sample `x`: `e_A(x) = arithmetic_mean_k e_k(x)` over all five
P3 Track-G checkpoints `k` of that arm. Implemented in
`synthetic_real_probe.average_checkpoint_evidence`. All five training seeds
contribute; none is selected. The detector TRAINING seed (which of the five
checkpoints) is never confused with the BA_sep PROBE seed (§5) — they are
different axes, bound separately in the protocol.

## 5. Probe seeds

```
probe_seed_values: [20260806, 20260807, 20260808]
```

**Provenance, exactly as required:** these numeric values intentionally reuse
the first three values of the Version-C training-seed family for
reproducibility, but their role here is NEWLY PREREGISTERED BA_sep PROBE
seeds — §18.3 did not originally fix them as probe seeds (confirmed in the
decision dossier's §D: §18.3 scopes that policy to hypothesis TRAINING rows).
No seed may be replaced or added after a BA result exists.

## 6. Populations

- **`real_spoof_population`:** canonical `source_train` real-spoof samples
  only, resolved via `prism_fas.data.loader.loose_dataset.CanonicalPackageDataset`
  (`source_train` split) — the same reader `M9TrainingDataset._real_pools`
  uses.
- **`synthetic_population`:** the frozen C6 matched bank for the ARM under
  probe, traced to the exact resolver C7/C8 already use:
  `prism_fas.detector.c6_bank.open_arm_bank`, called with the arm's
  `c6_evidence.verify_c6_evidence(repo).bank(arm)` evidence — the identical
  call `src/prism_fas/pipeline/adapters/c8.py:1346-1351` makes before
  training a row. No second resolver was written.
- **`source_domains`:** `[casia_fasd, msu_mfsd]` exactly.
- **Forbidden:** `source_dev` as a probe fitting population, SiW-Mv2, any
  target split, any target metadata.

## 7. Sample unit

One canonical 224×224 detector input crop. Never a video, subject, recipe or
checkpoint aggregate. Checkpoint evidence is averaged per §4; the statistical
sample stays one crop.

## 8. Matched source split

Deterministic group-stratified 80/20 split, computed BEFORE fitting the
probe:

```
sha256(split_hash_namespace + "|" + probe_seed + "|" + source_domain + "|" +
       stable_group_identity)
```

`split_hash_namespace = "c9-ba-sep-option1-v1"`. Group-safe (the same stable
identity always lands in the same bucket for a given seed/domain), shared
across arms (the split never depends on which arm's synthetic pool is
paired), and stable-identity-based — never file path or array index
(`sample_id` for real records, `synthetic_id` for synthetic records; both are
the canonical stable identities their respective readers already expose).
Implemented in `synthetic_real_probe.split_bucket` / `assign_splits`, unit
tested for determinism, group-safety and the ~80/20 fraction.

## 9. Class / domain balancing

1:1 real-vs-synthetic within each `(probe_seed, source_domain, split)` cell.
`N = min(real_spoof_available, RND_synthetic_available, DET_synthetic_available,
LLM_synthetic_available)`, computed per cell. The SAME selected real-spoof
subset is reused for all three arms. Selection order is deterministic SHA-256
order over `(protocol_identity, probe_seed, split, source_domain,
stable_group_identity)`. No replacement, no oversampling, no class weights.
Implemented in `synthetic_real_probe.balance_classes`.

## 10. Preprocessing

Identical to the frozen M9/C8 Track-G detector inference preprocessing. No
probe-specific image pipeline exists — the probe consumes detector evidence,
not raw pixels.

## 11. Feature normalization

Z-score, fit on the probe TRAIN split only (`fit_normalization`), applied to
validation with the frozen train statistics (`apply_normalization`). Never
fit jointly across arms. No PCA, no feature selection.
`epsilon = 1.0e-8`.

## 12. Linear probe

`torch.nn.Linear(2, 1)`, zero-initialized (weight and bias). No scikit-learn
was added — sklearn is not a project dependency and none was introduced.
`BCEWithLogitsLoss`, no class weighting. L2 on weight only, `lambda = 1e-4`.
Full-batch `torch.optim.LBFGS`:

```
lr = 1.0, max_iter = 200, max_eval = 250, tolerance_grad = 1e-7,
tolerance_change = 1e-9, history_size = 100, line_search_fn = "strong_wolfe"
```

CPU, `float64`. No minibatching, no validation-metric early stopping, no
hyperparameter/C/regularization/threshold search. Termination is only
through the frozen LBFGS numeric convergence tolerances above. Deterministic
given fixed data — proven by a regression that fits the same synthetic
fixture data twice and asserts identical resulting weights.

## 13. Prediction, metric, hard gate

`sigmoid(logit) >= 0.5` ⇒ synthetic; threshold fixed at `0.5`, no threshold
search. `prism_fas.train.metrics.balanced_accuracy` (already implemented,
`src/prism_fas/train/metrics.py:31-34`, no new dependency). `BA_sep_arm_seed`
per seed on that seed's held-out validation split; `BA_sep_arm` = arithmetic
mean over the three probe seeds. Hard gate unchanged: `BA_sep_arm <= 0.75`.
Not altered regardless of any future observed result.

## 14. C-H4 remains separate

`C_H4_SUPPORT_RULE` in `detector_reliability.py` is unchanged by this task. A
reliability PASS does not prove C-H4; negative evidence, if any is ever
produced, is retained unchanged.

## 15. Protocol config and identity

`configs/evaluation/c9_detector_ba_sep_option1_v1.yaml` — every
`PROBE_PROTOCOL_REQUIRED_FIELDS` entry plus `checkpoint_set_policy`,
`checkpoint_evidence_aggregation`, `split_hash_namespace`,
`validation_fraction`, `classifier_threshold`, `normalization_epsilon`-scoped
fields, the optimizer settings and `dtype`. `status: FROZEN_NOT_RUN`.

`prism_fas.evaluation.detector_reliability.protocol_identity(protocol)` —
sha256 over the canonical (sorted-key) serialization of every field EXCEPT
`frozen_on`, `approved_by`, `status`, `no_ba_sep_observed_before_freeze`,
`not_resolved_by_this_freeze`, `schema_version`, `decision_id`,
`resolves_test` (provenance/metadata, never result-affecting). Proven, by
regression, to change if and only if a result-affecting field changes, and to
stay identical when only metadata changes.

`detector_reliability.probe_protocol_status(repo)["resolved"]` is now `True`
— ONLY because `load_probe_protocol` found this exact frozen config file with
every required field bound. `DETECTOR_BA_SEP_PROBE_PROTOCOL` (the module
constant) stays `None`; no scientific value is frozen by editing a Python
literal in this project, matching every other frozen Version-C decision
(`c7_decision.load_decision`, `lr_decision.load_decision`). **Protocol
resolved is not the same as test result PASSED** — `barrier_state({})`
still reports `synthetic_vs_real_spoof_probe: UNRESOLVED`, and
`verify_lock()` still refuses (proven by regression).

## 16. Implementation

`src/prism_fas/evaluation/synthetic_real_probe.py` — protocol loading and
identity; checkpoint binding (`resolve_checkpoint_set`,
`resolve_all_checkpoint_sets`); population resolution (`resolve_real_spoof_population`,
`resolve_synthetic_population`); the deterministic split
(`split_bucket`/`assign_splits`); class balancing (`balance_classes`);
evidence extraction (`extract_evidence`, `forward_checkpoint_evidence`,
`average_checkpoint_evidence`); normalization (`fit_normalization`,
`apply_normalization`); the linear probe (`fit_linear_probe`,
`predict_probability`); BA_sep computation and aggregation
(`compute_ba_sep_for_seed`, `aggregate_ba_sep`); and a read-only
`preflight(repo)`.

**`run_scientific_probe(repo, arm)` deliberately raises `NotImplementedError`.**
Every resolver and every numeric mechanism it would call is implemented and
tested; the one missing piece — loading real checkpoint weights into a
`PRISMDetector` and forwarding real image batches through it — is exactly the
construction `M9Trainer`/`M9TrainingDataset`/`M9ValidationDataset` already do
for C8's own rows and cross-source diagnostics
(`src/prism_fas/pipeline/adapters/c8.py::_run_scientific_row`,
`_cross_source_evaluation`). Wiring it here, unverifiable on a machine with
no real checkpoints and no real source package, would be exactly the kind of
untested integration code this project's conventions refuse to ship. A
separate scientific runner, built on the GPU host, is the correct place for
it.

`src/prism_fas/evaluation/detector_reliability.py` — added `load_probe_protocol`,
`PROBE_PROTOCOL_CONFIG_PATH`, made `probe_protocol_identity` (renamed
`protocol_identity`) public, and extended `probe_protocol_status` to accept
an optional `repo` (backward compatible: no-arg behavior is byte-identical to
before).

## 17. Tests

`tests/pipeline/test_c9_ba_sep_option1_protocol.py` — 44 tests covering every
item in the freeze task's required list: protocol completeness and identity
determinism/sensitivity; the exact 2-D evidence vector and that no forbidden
field is ever read; the 15-checkpoint requirement and refusal on partial
sets; arithmetic-mean checkpoint aggregation; the exact three probe seeds;
deterministic, group-safe, cross-arm-shared splitting; 1:1 balance with a
shared real subset; train-only normalization; zero-initialized, frozen-solver
linear probe fitting (deterministic, regression-tested); the fixed 0.5
threshold; BA aggregation over exactly three seeds; the 0.75 ceiling; the
target firewall; that `preflight` never fits a probe or computes BA; that
importing the module has no side effects; that `run_scientific_probe` is not
wired; that the module never writes a file; and that `verify_lock` still
blocks — both generally and specifically when only one of the nine required
tests would be PASSED.

All 44 pass. Existing `test_detector_reliability_stage.py` (54 passed, 1
skipped, unchanged), `test_c6_scientific_contract.py`, and
`test_c9_scientific_evidence.py` remain green.

## 18. What this freeze does NOT do

- Does not compute a real BA_sep value.
- Does not resolve the other eight `REQUIRED_DETECTOR_RELIABILITY_TESTS`.
- Does not touch `crop_padding_interpolation`'s data-blocked status.
- Does not create `DETECTOR_RELIABILITY_LOCK_C.json`.
- Does not run C9, C10, C11, C12 or C13.
- Does not access target data. `target_access = 0` throughout.
- Does not retrain C7 or any C8 row, add `C-R-RND`, or alter any C7 winner.

**C9 remains correctly BLOCKED after this task.** That is expected —
resolving one test's protocol out of nine required tests cannot and does not
make `detector_reliability.verify_lock()` valid.
