# C7-C13 production-path readiness

**Engineering audit artifact. Not scientific evidence.** Nothing here is a
measurement, and nothing here may support a claim. It answers one question
per stage: could `--profile full` produce a scientific result today, and if
not, exactly what stops it.

Generated `2026-08-24T10:03:25.898225Z` on development laptop; no CUDA, no source package, no target package.
`target_access = 0` — no target path, label or
metric was resolved while producing it.

The three structural facts per stage — does a `_scientific_workflow` exist,
does `workflow` dispatch on the execution context, and is every fixture
producer guarded — are read out of the source by
`scripts/audit_c7_c13_readiness.py`, so this table cannot drift from the
repository the way a hand-written one does.

## Summary

| | stages |
| --- | --- |
| scientific workflow present | C7, C8, C9 |
| no scientific workflow | C10, C11, C12, C13 |
| code path ready | C7, C8 |
| runnable on THIS host | — |
| unguarded fixture call sites | — |
| unresolved scientific decisions | C9 |
| scientific path ever executed | — |

## C7 — Detector readiness and configuration search

**Purpose.** prove every typed primary row is executable on a CPU fixture, enforce the §13.5 decision-dependency guards, and close the §15.2.2 detector/loss envelope in ONE bounded coordinate pass before C8

- module: `src/prism_fas/pipeline/adapters/c7.py`
- modes: `TRACK_G_READINESS`, `TRACK_R_READINESS`, `DECISION_DEPENDENCY_AUDIT`, `CALIBRATION_GUARDS`, `VARIANT_MATRIX_AUDIT`, `SOURCE_SEARCH`, `VERIFY_C6_EVIDENCE`, `SCIENTIFIC_SOURCE_SEARCH`, `FINALIZE_DETECTOR_CONFIG`, `VERIFY_CONFIG_LOCK`
- scientific substages: `VERIFY_C6_EVIDENCE`, `SCIENTIFIC_SOURCE_SEARCH`, `FINALIZE_DETECTOR_CONFIG`, `VERIFY_CONFIG_LOCK`
- requires an accelerator for scientific execution: **True**
- engineering/rehearsal path: **True**
- scientific path implemented: **True**
- `workflow` dispatches on the context: **True**
- declares semantic preconditions beyond existence: **True**
- may claim scientific evidence: **True**
- produces lock: `reports/full/c7/DETECTOR_CONFIG_LOCK.json`
- lock verifier: `prism_fas.pipeline.adapters.c7.verify_detector_config_lock (module level; shared with C8)`
- target capability required: none
- target capability forbidden: all
- scientific path ever executed: **False**
- **code path ready: True** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 3 absent input(s), and needs an accelerator)

### Required inputs

| input | path | present here |
| --- | --- | --- |
| detector_config | `configs/train/m9_reference.yaml` | yes |
| detector_model_config | `configs/models/m9_detector.yaml` | yes |
| pretrained_weights | `weights` | yes |
| source_package | `data/packages/prism_data_v1_m3b` | **no** |
| c6_matched_banks | `reports/full/c6` | **no** |
| c5_candidates | `runs/full/c5/scientific/candidates` | **no** |

### Hard acceptance

- Track G and Track R typed variants instantiate, forward, produce a finite loss, backward, step, checkpoint and resume on a CPU fixture
- no experiment-id branching; both tracks are configurations of one implementation
- Track-R ConvNeXt and RegionFusion have non-zero autograd dependency on the fused logit, and the branch-intervention audit passes
- Track G decides on global_logit_G / p_G; Track R on fused_logit_R / p_R
- decision_graph_hash serialized into run identity
- manifold=OFF primary Track R executes no L_real / L_out / L_clean
- K=4 is an explicit typed secondary variant
- calibration fits and thresholds the SAME decision quantity
- one deterministic coordinate pass in the frozen §15.2.2 order, candidates anchor x {0.5,1,2} (warm-up x {0.5,1,1.5} clipped to [0,0.20]), inactive terms skipped, every trial retained, canonical SHA-256 tie-break
- DETECTOR_CONFIG_LOCK.json written only after the envelope is terminal

### Fixture call sites

| function | producers | guarded |
| --- | --- | --- |
| `_decision_audit` | `_fixture_batch`, `build_audit_detector` | yes |
| `_fixture_batch` | `audit_batch` | yes |
| `_track_readiness` | `_fixture_batch`, `build_audit_detector` | yes |

### Resolved scientific decisions

- C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET, FROZEN 2026-08-24 before any C7 scientific metric existed. One bounded pass per TRACK, both anchored on the C6 DET bank; every primary generator arm of a track trains at that track's single frozen configuration in C8. Record: configs/search/c7_source_search_decision.yaml, identity ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
- the learning-rate anchor interpretation (B_common_multiplier), APPROVED in configs/search/lr_anchor_decision.yaml
- the optimizer family, uniquely inherited as AdamW from configs/train/m9_reference.yaml
- the protocol (P3), the ranking tuple (P3_READY) and the per-trial schedule (frozen_m9_schedule), all in the same decision record

### Unresolved result-affecting decisions

none.

### Blockers

- reports/full/c6 absent on this host (the C6 GPU evidence lives on the execution backend)
- data/packages/prism_data_v1_m3b absent
- runs/full/c5/scientific/candidates absent
- weights/ pinned SigLIP2 + ConvNeXt absent
- no CUDA device

### Code paths still unexercised

- the real M9Trainer flow inside a C7 trial (rehearsed with a stub trainer; the trainer itself is covered by tests/test_m9_regional_detector)
- C6MatchedBankReader over 1024 real candidates (rehearsed at 4)
- the CUDA branch of _scientific_device

**Safe to run:** NO on this laptop — the scientific path requires CUDA, the M3B package, the pinned weights, the C5 candidate tree and the frozen C6 closure, none of which is here. Every result-affecting DECISION is now frozen, so what remains is inputs and hardware. The engineering readiness path runs on CPU.

## C8 — Source matrix over arms, tracks, configs and seeds

**Purpose.** execute the frozen §18 source-only matrix — 42 atomic runs over protocol x method x config x seed — at the configuration C7 froze, and emit durable per-run evidence for every one

- module: `src/prism_fas/pipeline/adapters/c8.py`
- modes: `PLAN_MATRIX`, `SCHEDULE`, `EXECUTE_ROWS`, `FAILURE_PRESERVATION`, `TARGET_ISOLATION`, `VERIFY_INPUTS`, `CROSS_SOURCE_DIAGNOSTICS`, `CALIBRATION_STABILITY`, `ACCEPTANCE`
- scientific substages: `VERIFY_INPUTS`, `CROSS_SOURCE_DIAGNOSTICS`, `CALIBRATION_STABILITY`, `ACCEPTANCE`
- requires an accelerator for scientific execution: **True**
- engineering/rehearsal path: **True**
- scientific path implemented: **True**
- `workflow` dispatches on the context: **True**
- declares semantic preconditions beyond existence: **True**
- may claim scientific evidence: **True**
- produces lock: `reports/full/c8/C8_ACCEPTANCE.json`
- lock verifier: `prism_fas.pipeline.adapters.c8.C8Adapter._scientific_acceptance, over the canonical plan`
- target capability required: none
- target capability forbidden: all
- scientific path ever executed: **False**
- **code path ready: True** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 4 absent input(s), and needs an accelerator)

### Required inputs

| input | path | present here |
| --- | --- | --- |
| c6_matched_banks | `reports/full/c6` | **no** |
| c7_config_lock | `reports/full/c7/DETECTOR_CONFIG_LOCK.json` | **no** |
| source_packages | `data/packages` | **no** |
| pretrained_weights | `data/packages/pretrained` | **no** |

### Hard acceptance

- Track-G P1/P2: RND/DET/LLM at 3 seeds each
- Track-G P3-ready: RND/DET/LLM at 5 seeds each
- Track-R P3-ready: DET/LLM at 3 seeds each
- PromptHead LLM ON/OFF ablation at 3 seeds
- 42 atomic rows, each with durable evidence; no hidden row, no omitted row
- P1/P2 checkpoint selection on that protocol's source_dev only; the cross-domain side is diagnostic and never a selection signal
- P3-ready selection on equal-weight CASIA-dev + MSU-dev, never SiW
- calibration on source_dev only, fitting and thresholding the row's own decision logit and score
- cross-source diagnostics and calibration stability exist before C9
- no SiW P3 scoring

### Fixture call sites

| function | producers | guarded |
| --- | --- | --- |
| `_run_one` | `audit_batch`, `build_audit_detector` | yes |

### Resolved scientific decisions

- the §18 matrix composition and the fixed seed family 20260806-20260810, materialized by prism_fas.evaluation.source_matrix.build_plan

### Unresolved result-affecting decisions

none.

### Blockers

- reports/full/c7/DETECTOR_CONFIG_LOCK.json absent — C7's decisions are all frozen but C7 has not RUN scientifically, so the lock does not exist yet
- the same absent C6/C5/package/weight inputs as C7
- no CUDA device

### Code paths still unexercised

- the real M9Trainer flow inside a row (rehearsed with a stub trainer)
- _cross_source_evaluation over a real second M9ValidationDataset
- the full 42-row schedule (4 representative rows rehearsed)

**Safe to run:** NO on this laptop — blocked on the same missing inputs as C7 plus a verifying DETECTOR_CONFIG_LOCK. The rehearsal path runs on CPU over a bounded sample.

## C9 — Source matrix freeze

**Purpose.** freeze SOURCE_MATRIX_LOCK_C over the completed source matrix, or refuse and name every reason

- module: `src/prism_fas/pipeline/adapters/c9.py`
- modes: `BUILD_LOCK`, `VALIDATE_LOCK`, `REFUSAL_CASES`, `LOAD_C8_EVIDENCE`, `FREEZE_SOURCE_MATRIX`
- scientific substages: `LOAD_C8_EVIDENCE`, `FREEZE_SOURCE_MATRIX`, `VALIDATE_LOCK`
- requires an accelerator for scientific execution: **False**
- engineering/rehearsal path: **True**
- scientific path implemented: **True**
- `workflow` dispatches on the context: **True**
- declares semantic preconditions beyond existence: **True**
- may claim scientific evidence: **True**
- produces lock: `reports/full/c9/SOURCE_MATRIX_LOCK_C.json`
- lock verifier: `prism_fas.evaluation.source_lock.validate`
- target capability required: none
- target capability forbidden: all
- scientific path ever executed: **False**
- **code path ready: False** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 3 absent input(s))

### Required inputs

| input | path | present here |
| --- | --- | --- |
| c8_runs | `runs/full/c8` | **no** |
| c8_acceptance | `reports/full/c8/C8_ACCEPTANCE.json` | **no** |
| detector_reliability_lock | `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json` | **no** |

### Hard acceptance

- all mandatory C8 rows complete and terminal
- no failed and no hidden row
- every checkpoint, calibration and run identity frozen
- DETECTOR_RELIABILITY_LOCK_C valid

### Fixture call sites

| function | producers | guarded |
| --- | --- | --- |
| `_build` | `_complete_evidence` | yes |
| `_refusals` | `_complete_evidence` | yes |
| `_validate` | `_complete_evidence` | yes |

### Resolved scientific decisions

- the staging of the synthetic-vs-real barrier: after C8, before C9 (SYNTHETIC_VS_REAL_RELIABILITY_STAGE, frozen 2026-08-23)

### Unresolved result-affecting decisions

- DETECTOR_BA_SEP_PROBE_PROTOCOL — NEEDS_SCIENTIFIC_DECISION
- DETECTOR_BA_SEP_EVIDENCE_VECTOR — NEEDS_SCIENTIFIC_DECISION
- DETECTOR_BA_SEP_PROBE_SEEDS — NEEDS_SCIENTIFIC_DECISION

### Blockers

- DETECTOR_RELIABILITY_LOCK_C absent and its protocol unfrozen
- C8 has not run scientifically

### Code paths still unexercised

- the freeze over 42 real C8 manifests (rehearsed over 42 synthetic ones written in C8's own manifest schema)

**Safe to run:** NO — and correctly so. C9 BLOCKS on DETECTOR_RELIABILITY_LOCK_C, whose protocol is unfrozen. The barrier must be decided separately; it may NOT be chosen from C8 outcomes.

## C10 — Target package and label firewall

**Purpose.** build and lock the sealed SiW-Mv2 target package and the label firewall, without opening a label

- module: `src/prism_fas/pipeline/adapters/c10.py`
- modes: `BUILD_FIXTURE_PACKAGE`, `FIREWALL_PERMISSIONS`, `PACKAGE_IDENTITY`, `TARGET_LOCK`, `TAMPER_DETECTION`
- scientific substages: none
- requires an accelerator for scientific execution: **False**
- engineering/rehearsal path: **single workflow**
- scientific path implemented: **False**
- `workflow` dispatches on the context: **False**
- declares semantic preconditions beyond existence: **False**
- may claim scientific evidence: **False**
- produces lock: `reports/full/c10/TARGET_PACKAGE_LOCK.json (rehearsal-only today)`
- lock verifier: `prism_fas.evaluation.firewall.TargetLabelFirewall`
- target capability required: sealed_real (scientific only)
- target capability forbidden: labels, under every profile; features under any non-eligible profile
- scientific path ever executed: **False**
- **code path ready: False** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 2 absent input(s))

### Required inputs

| input | path | present here |
| --- | --- | --- |
| target_evaluation_config | `configs/evaluation/m10_target.yaml` | yes |
| c9_source_lock | `reports/full/c9/SOURCE_MATRIX_LOCK_C.json` | **no** |
| target_feature_package | `data/processed/prism_target_eval_v2` | **no** |

### Hard acceptance

- the target feature package is mounted READ-ONLY for C11
- no training/LLM/synthesis stage may resolve a label root
- package identity verified and tamper-detectable

### Fixture call sites

none.

### Unresolved result-affecting decisions

none.

### Blockers

- reports/full/c9/SOURCE_MATRIX_LOCK_C.json absent
- data/processed/prism_target_eval_v2 absent
- no scientific workflow is written; the stage is rehearsal-only

### Code paths still unexercised

- every scientific target-package path

### Fail-closed defect fixed in this task

`_build_fixture` called `sources.target_roots(...)` and then `mkdir`-ed every returned root and wrote a `labels.json` of INVENTED labels into it. Under a scientific context that call returns the REAL sealed package roots, so the stage would have written fabricated labels inside the artifact the firewall exists to protect. Now guarded by `assert_fixture_permitted` before any path is created.

**Safe to run:** NOT in this task. Dry structural validation only; no target bytes were opened.

## C11 — Label-isolated P3 prediction

**Purpose.** run label-isolated P3 prediction over the sealed target package

- module: `src/prism_fas/pipeline/adapters/c11.py`
- modes: `BUILD_PREDICTIONS`, `LABEL_ISOLATION_AUDIT`, `PREDICTION_LOCKS`, `DOUBLE_VALIDATION`
- scientific substages: none
- requires an accelerator for scientific execution: **True**
- engineering/rehearsal path: **single workflow**
- scientific path implemented: **False**
- `workflow` dispatches on the context: **False**
- declares semantic preconditions beyond existence: **False**
- may claim scientific evidence: **False**
- produces lock: `reports/full/c11/TARGET_PREDICTION_LOCKSET.json (rehearsal-only today)`
- lock verifier: `prism_fas.evaluation.target_prediction.validate_predictions`
- target capability required: features, read-only
- target capability forbidden: labels
- scientific path ever executed: **False**
- **code path ready: False** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 3 absent input(s), and needs an accelerator)

### Required inputs

| input | path | present here |
| --- | --- | --- |
| c9_source_lock | `reports/full/c9/SOURCE_MATRIX_LOCK_C.json` | **no** |
| c10_target_lock | `reports/full/c10/TARGET_PACKAGE_LOCK.json` | **no** |
| target_feature_package | `data/processed/prism_target_eval_v2` | **no** |

### Hard acceptance

- no prediction row carries a ground-truth label, attack family, raw path, subject/session taxonomy or hidden target metadata
- each row's PREDICTION_LOCK binds its checkpoint, calibration, inference config and package identity
- the TARGET_PREDICTION_LOCKSET is validated twice before label capability is granted

### Fixture call sites

| function | producers | guarded |
| --- | --- | --- |
| `_build` | `prediction_rows` | yes |

### Unresolved result-affecting decisions

none.

### Blockers

- C9 and C10 have not run scientifically
- no scientific workflow is written; prediction rows come from adapters.tiny under rehearsal

### Code paths still unexercised

- every scientific target-inference path

### Fail-closed defect fixed in this task

`_build` constructed prediction rows from `tiny.prediction_rows` with no context guard, so a scientific context would have written invented scores behind a PREDICTION_LOCK. Now guarded.

**Safe to run:** NOT in this task. No target inference was executed.

## C12 — Scoring, statistics and hypothesis tests

**Purpose.** unlock the sealed labels inside the isolated C-G8 scorer and compute the P3 statistics

- module: `src/prism_fas/pipeline/adapters/c12.py`
- modes: `SCORER_ISOLATION`, `DRY_RUN`, `UNLOCK_AND_SCORE`, `STATISTICS`, `NO_FEEDBACK`
- scientific substages: none
- requires an accelerator for scientific execution: **False**
- engineering/rehearsal path: **single workflow**
- scientific path implemented: **False**
- `workflow` dispatches on the context: **False**
- declares semantic preconditions beyond existence: **False**
- may claim scientific evidence: **False**
- produces lock: `none`
- lock verifier: `prism_fas.evaluation.scoring (isolated C-G8 scorer)`
- target capability required: labels, read-only, scorer-scoped
- target capability forbidden: any label access outside the scorer; any write to model state or calibration
- scientific path ever executed: **False**
- **code path ready: False** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 2 absent input(s))

### Required inputs

| input | path | present here |
| --- | --- | --- |
| c11_lockset | `reports/full/c11/TARGET_PREDICTION_LOCKSET.json` | **no** |
| target_labels | `data/evaluation_only/prism_target_v2_labels` | **no** |

### Hard acceptance

- the scorer's import closure contains no training capability
- a dry run validates preconditions without opening label bytes
- label capability is refused before a lockset exists
- nothing C12 writes can mutate a C0-C11 artifact
- a single-seed comparison is refused rather than reported

### Fixture call sites

| function | producers | guarded |
| --- | --- | --- |
| `_dry_run` | `evaluation_labels` | yes |
| `_score` | `evaluation_labels` | yes |
| `_statistics` | `evaluation_labels` | yes |

### Unresolved result-affecting decisions

none.

### Blockers

- C11 has not run scientifically
- no scientific workflow is written; labels are fabricated from video ids by adapters.tiny under rehearsal

### Code paths still unexercised

- the real sealed-label resolution path

### Fail-closed defect fixed in this task

`workflow` scored fabricated labels with no context guard. Now guarded, so a scientific profile cannot score invented labels.

**Safe to run:** NOT in this task. No label capability was granted and no label byte was opened.

## C13 — Acceptance, evidence package and report

**Purpose.** final acceptance, the evidence package, and the refusal to declare completion while upstream milestones are incomplete

- module: `src/prism_fas/pipeline/adapters/c13.py`
- modes: `ACCEPTANCE_MATRIX`, `NEGATIVE_PRESERVATION`, `ARTIFACT_INTEGRITY`, `CLAIM_POLICY`, `FINAL_REPORT`
- scientific substages: none
- requires an accelerator for scientific execution: **False**
- engineering/rehearsal path: **single workflow**
- scientific path implemented: **False**
- `workflow` dispatches on the context: **False**
- declares semantic preconditions beyond existence: **False**
- may claim scientific evidence: **False**
- produces lock: `reports/full/c13/C_ACCEPTANCE.json (never produced)`
- lock verifier: `none`
- target capability required: none
- target capability forbidden: all
- scientific path ever executed: **False**
- **code path ready: False** — a statement about the implementation, not about this host
- **runnable on this host: False** (blocked on 1 absent input(s))

### Required inputs

| input | path | present here |
| --- | --- | --- |
| c12_statistics | `reports/full/c12/C12_ACCEPTANCE.json` | **no** |
| master_run_index | `state/MASTER_RUN_INDEX.json` | yes |

### Hard acceptance

- the acceptance matrix assembles over real upstream status
- negative and blocked results survive into the evidence package
- artifact integrity is checked by re-hashing, not by trusting a manifest
- a superiority claim with no statistical support is rejected
- C13 proposes a tag; it never creates the scientific tag

### Fixture call sites

none.

### Unresolved result-affecting decisions

none.

### Blockers

- C4-C12 have not run scientifically

### Code paths still unexercised

- C_ACCEPTANCE under a complete pipeline

**Safe to run:** NOT as science. C13's honest verdict today is a refusal naming C4-C12 as scientifically incomplete.
