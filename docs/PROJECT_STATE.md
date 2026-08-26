# PROJECT_STATE

Derived handoff state. **Not scientific authority** — the spec, immutable locks and Git
artifacts are. Every field below was verified during the v1.5 reconciliation task.

```yaml
project: PRISM-FAS-C-LLM

authoritative_spec:
  path: docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx
  sha256: ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5
  read_in_full: true
  canonical_location_per_spec_M1: docs/specs/   # deviation recorded; move planned

repository:
  branch: portable-one-command-full-run
  previous_branch: pre-gpu-scientific-decision     # LR decision dossier at 758fbe4
  branch_point: 36f10fd24880a0bfb3e6c3c2ba8a3fcc53195572   # the accepted C3 scientific checkpoint
  latest_accepted_checkpoint: 36f10fd24880a0bfb3e6c3c2ba8a3fcc53195572   # C3 banks frozen
  origin: https://github.com/COngtuthien/PRISM_FAS_C_LLM_Project.git
  version_b_remote_push: DISABLED_NO_PUSH_TO_VERSION_B

version_b:
  repo_path: D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project
  head: 7799f7decd35db6987ce4578824e5bd8d9eab4ae
  tag: m10-blind-evaluation-checkpoint
  tag_peeled_commit: 7799f7decd35db6987ce4578824e5bd8d9eab4ae
  clean: true
  immutable_verified: true

current_milestone: C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_PREEXECUTION_CORRECTION
current_substage: >-
  PRE-EXECUTION SCIENTIFIC CORRECTION of C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1.
  V1 (`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`, identity
  `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`) was
  NEVER scientifically executed
  (`no_diagnostic_metric_observed_before_freeze: true`; every mode of its
  runner, run once against this real repo, correctly reported an unresolved
  precondition and wrote nothing) — so this is a pre-execution audit
  correction, not a revision of an observed result. V1 is preserved
  byte-for-byte, unchanged, as historical pre-execution design evidence
  (confirmed by `git diff --stat HEAD` and by protocol-identity regression);
  it is never edited and V2 never writes into its artifact namespace. A
  pre-execution scientific audit found five defects in V1's design, all
  corrected in a new, separately identified protocol,
  `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2`
  (`configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml`, identity
  `05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523`):
  (A) the benign-corruption threshold was self-normalizing (derived from the
  tested corruption's own calibration-group shift) — corrected to a disjoint
  reference-calibration design using the pre-existing, independently-frozen
  M8 benign population (`synthesis.quality_calibration.BENIGN_VARIANTS`/
  `BENIGN_NOISE_STD`), forwarded on the calibration group only, scored
  against `tau_mean`/`tau_tail` derived from it, while the tested corruption
  is forwarded only on the evaluation group; (B) `cross_route_synthetic`
  reused the BA_sep separability ceiling (LOW is good) under a test whose
  canonical declared pass_rule is "performance is retained across routes"
  (HIGH is good) — reclassified `NEEDS_SCIENTIFIC_DECISION`, V1's attempt
  preserved unused; (C) an existing complete result set was validated too
  weakly before re-reporting — `validate_existing_diagnostics_result` is now
  canonical and gates every second-or-later `--execute` and every `--status`
  call, blocking (exit 2) rather than recomputing or overwriting an invalid
  complete result; (D) the C8 matrix identity was handwritten placeholder
  text — now the real `source_matrix.build_plan().identity`, cross-checked
  against `C8_ACCEPTANCE.json`, fail-closed on mismatch or absence; (E) the
  group-safe calibration/evaluation split was proven only globally — now
  proven per domain (`casia_fasd`, `msu_mfsd` independently). Executable-test
  count DROPS from V1's 4 to V2's 3 (`benign_jpeg_corruption`/
  `benign_resize_corruption`/`benign_color_corruption` remain executable;
  `cross_route_synthetic` no longer is) — the correction removes a test from
  the executable set rather than inventing a metric to keep it executable.
  `python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner
  {--preflight-only|--bind-only|--status|--execute}` mirrors V1's runner
  contract on its own separate namespace
  (`reports/full/c8/reliability/post_failure_source_diagnostics_v2/`, proven
  by regression to never touch V1's namespace, `.../synthetic_vs_real_spoof_probe/`,
  or `DETECTOR_RELIABILITY_LOCK_C.json`). NOTHING WAS RUN FOR REAL:
  `--preflight-only`, `--bind-only`, `--status` and `--execute` were each run
  once against this real repo on this laptop (no CUDA, no M3B package, no
  `C8_ACCEPTANCE.json`) and each correctly reported an unresolved
  precondition, exit 2, writing nothing. `synthetic_vs_real_spoof_probe`
  remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
  `overall=FAILED`, `c9_may_close=false`; C9 remains
  `BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE`. `target_access` has been 0
  throughout. See `stage_table.c9.post_failure_source_diagnostics_v2` below
  for full detail and `stage_table.c9.post_failure_source_diagnostics_v1`
  for the (unchanged, superseded-before-execution) prior state.
_superseded_current_substage_v1: >-
  SUPERSEDES the substages below. On the GPU host, C4, C5, C6, C7 and C8 are
  SCIENTIFIC CLOSED. `synthetic_vs_real_spoof_probe` has scientifically FAILED
  under `C9_DETECTOR_BA_SEP_OPTION1_V2` (BA_sep_RND=0.7843079833902619,
  BA_sep_DET=0.8514170182841069, BA_sep_LLM=0.7902658339472685, all over the
  0.75 ceiling) — see the prior milestone and
  `reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.md`. The
  GPU host has now ALSO registered the failed barrier (user-reported):
  `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`, `identity_sha256 =
  40825a5fffcbbdd681e5d8b0354e8371dcccabc36b51d1b9a12cd8bd29e73fbe`,
  `registered=true, reused=false, overall=FAILED, c9_may_close=false,
  target_access=0`; `validate_lock_record` reports `valid=true` (a genuine
  structurally-valid FAILED record); `verify_lock` correctly reports
  `valid=false` (`overall is FAILED, not PASSED`, etc. — unweakened). THIS
  LAPTOP HAS NOT INDEPENDENTLY READ THESE GPU FILES; the registration is
  OBSERVED_ON_GPU_USER_REPORTED, not GPU_ARTIFACT_REGISTRATION_VERIFIED by
  this laptop. This milestone follows the recommended Option B (bounded
  source-only diagnostics) from the post-failure decision dossier: it
  FREEZES a NEW, separately identified protocol,
  `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1`
  (`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`, identity
  `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`) and
  implements (never runs) it. NOT a BA_sep revision, NOT a reliability-
  barrier rescue, NOT a C9 pass path, NOT a target protocol — every artifact
  this protocol's runner would write hard-codes `c9_may_close: false` and
  the BA_sep observed verdict as `FAIL`. A code re-audit (detector
  contracts, the model forward pass, the loss module, C6/C7/synthesis
  structures) found 4 of the 8 remaining required tests genuinely
  `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL`
  (`benign_jpeg_corruption`/`benign_resize_corruption`/`benign_color_corruption`
  — source_dev LIVE only, a preregistered group-safe calibration/evaluation
  split derives the acceptance threshold, reusing `reliability.score_shift`;
  `cross_route_synthetic` — reuses the EXACT frozen BA_sep linear-probe
  mechanics, fit on one synthesis route's evidence and scored on the other,
  reusing the same 0.75 ceiling rather than inventing one), 2 more precisely
  reclassified `NEEDS_SCIENTIFIC_DECISION` with code-grounded reasons
  (`recipe_region_shift`, `artifact_map_swap` — MORE evidence now known
  available than a prior audit assumed — `region_embeddings`/`region_attention`/
  `local_logits`/`attack_region_mask` all genuinely exist for primary Track R —
  but which existing signal legitimately maps onto the spec's named
  quantity, and its acceptance threshold, remain undecided), 1
  `STRUCTURALLY_MODEL_BLOCKED` (`residual_scale_zero` — needs synthesis-time
  residual control this task may not exercise, and `region_distances`, which
  no primary Version-C row's checkpoint produces), and 1 re-confirmed
  `STRUCTURALLY_DATA_BLOCKED` (`crop_padding_interpolation` — the frozen
  package retains no bounding box or source-frame path). A dedicated runner,
  `python -m prism_fas.evaluation.post_failure_diagnostics_runner
  {--preflight-only|--bind-only|--status|--execute}`, mirrors the BA_sep
  runner's exact no-rerun/partial-block contract on an entirely separate
  artifact namespace
  (`reports/full/c8/reliability/post_failure_source_diagnostics_v1/`,
  proven by regression to never touch
  `.../synthetic_vs_real_spoof_probe/` or `DETECTOR_RELIABILITY_LOCK_C.json`).
  NOTHING WAS RUN FOR REAL: `--preflight-only`, `--bind-only`, `--status`
  and `--execute` were each run once against this real repo on this laptop
  (no CUDA, no M3B package) and each correctly reported an unresolved
  precondition, exit 2, writing nothing. The BA_sep V2 protocol identity is
  unchanged (confirmed by regression); no BA_sep V3 was created; the BA_sep
  artifacts and `DETECTOR_RELIABILITY_LOCK_C.json` remain untouched
  (confirmed by direct byte-comparison tests). `verify_lock` remains strict
  and unweakened. C9 remains `BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE`. The
  post-failure EXPLORATORY TARGET protocol is explicitly NOT started —
  deferred to a separate, future task, only after these diagnostics are
  observed and frozen. target_access has been 0 throughout.
previous_milestone: C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_PROTOCOL_FREEZE
execution_profile: rehearsal   # THIS LAPTOP: `python train.py` still resolves
  # CPU_FULL_REHEARSAL here (no CUDA, no source package). The GPU host
  # separately ran --profile full through C8, ran the real BA_sep --execute
  # to an observed FAILED verdict, and has now separately (user-reported)
  # registered that FAILED verdict into DETECTOR_RELIABILITY_LOCK_C.json;
  # see execution_pipeline.full and
  # stage_table.c9.ba_sep_option1_v2_gpu_registration.
pipeline_phase: scientific-execution-through-c8-c9-blocked-by-registered-detector-reliability-failure-post-failure-diagnostics-frozen

# SCOPE WARNING — HISTORICAL, SUPERSEDED 2026-08-25. Kept for context: this is
# what was true through the C7_C13_PRODUCTION_PATH_READINESS milestone, when
# C4-C13 had adapters but no scientific execution. It no longer describes the
# current state (`scientific_status` below is no longer C3-only) — see
# `current_substage` above and `c7_scientific_closure_reconciliation` below
# for what is current.
#
# `engineering_status: SMOKE_PASS` is now scoped to the WHOLE C0-C13 pipeline: every
# stage has an adapter, and `--profile smoke --from C0 --to C13` traverses all
# fourteen and passes. That is an ENGINEERING statement about code paths.
#
# `scientific_status` was, AT THAT TIME, scoped to C3. C4-C13 had adapters and
# had STILL never executed scientifically. An adapter proves a stage CAN run;
# it is not evidence that it HAS. Read `execution_pipeline` and
# `historical_milestones` before quoting either field.
engineering_status_scope: C0_TO_C13_FULL_PIPELINE_ENGINEERING
engineering_status: SMOKE_PASS      # C0-C13, validate PASS + smoke PASS, 2026-08-17
# CURRENT, 2026-08-26: C3, C4, C5, C6, C7 and C8 are scientifically closed on
# the GPU host (frozen identities in c7_scientific_closure_reconciliation
# below; C8's 42/42-row closure recorded here and in
# reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md). C9 is
# BLOCKED_PENDING_DETECTOR_RELIABILITY_SCIENTIFIC_DECISION. C10-C13 are
# NOT_YET_SCIENTIFICALLY_EXECUTED. A single PASS/FAIL scalar cannot represent
# that split truthfully, so scientific_status is now a per-stage map.
scientific_status:
  c3: PASS
  c4: PASS
  c5: PASS
  c6: PASS
  c7: PASS
  c8: PASS
  c9: BLOCKED_PENDING_DETECTOR_RELIABILITY_SCIENTIFIC_DECISION
  c10_c13: NOT_YET_SCIENTIFICALLY_EXECUTED
scientific_status_scope: c3_through_c8_closed_c9_blocked_pending_reliability_decision_c10_c13_pending

# Machine-readable so no parser can infer these exist. Previously this fact lived
# only in YAML comments beside real-looking paths, which a parser strips.
execution_pipeline:
  validate: IMPLEMENTED_C0_TO_C13   # 14 stages, 21 checks, 0 failed
  smoke: IMPLEMENTED_C0_TO_C13      # 14 stages, 62 substage modes, 242 checks, 0 failed
  # CURRENT, 2026-08-26: live C3 generation ran 2026-08-16; the GPU host has
  # since run --profile full scientifically through C4, C5, C6, C7 and C8 too
  # (see c7_scientific_closure_reconciliation below). C9-C13 remain BLOCKED —
  # C9 on the detector-reliability protocol decision (USER_APPROVAL_REQUIRED,
  # reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md).
  full: EXECUTED_C3_THROUGH_C8
  # Separate axis, deliberately not folded into `full` above. The scientific CODE
  # PATH for C4-C13 now exists and C4-C8 of it has actually executed and closed;
  # C9-C13 have not yet executed scientifically. The refusing `run_full` is GONE —
  # one `workflow()` per stage, driven by an ExecutionContext, with no placeholder
  # to fall back to. On THIS LAPTOP every stage still BLOCKS on a named missing
  # input (no GPU, no source package) — that is a laptop-local fact, not a
  # statement about whether the GPU host has run them.
  full_path_implemented_c4_to_c13: true
  full_path_placeholders_remaining: 0
  full_path_ever_executed_c4_to_c13: PARTIAL_C4_THROUGH_C8_CLOSED_C9_C13_NOT_RUN
  orchestrator_exists: true     # train.py + src/prism_fas/pipeline/
  pipeline_state_exists: true   # state/PIPELINE_STATE.json
  master_run_index_exists: true # state/MASTER_RUN_INDEX.json
  stage_adapters_exist: C0_TO_C13_ALL
  c0_to_c13_pipeline_ever_run: ENGINEERING_AND_PARTIAL_SCIENTIFIC
  c0_to_c13_pipeline_ever_run_scientifically: PARTIAL_C3_THROUGH_C8
  note: >-
    Every C0-C13 stage has an adapter and every stage's control path executes and
    passes under validate and smoke. CURRENT, 2026-08-26: on the GPU host, C3
    through C8 have each executed LIVE under --profile full and are scientifically
    closed (see c7_scientific_closure_reconciliation below for identities and the
    C7/C8 closure verdicts). C9 through C13 have executed ONLY on fixtures on any
    host so far: under --profile full each of them still BLOCKS on an absent input
    or an unresolved decision and names it — C9 on
    DETECTOR_RELIABILITY_LOCK_C, whose executable protocol is
    USER_APPROVAL_REQUIRED (see
    reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md),
    C10-C13 on their own preconditions — and no
    scientific target inference or scoring has ever run. SMOKE_PASS over C0-C13
    means ENGINEERING_READY for the whole pipeline; scientific completion is now
    per-stage (see `scientific_status` above), not a single C0-C13 scalar.

# --- C4-C13 engineering readiness (this milestone) ---------------------------
c4_to_c13_engineering_readiness:
  status: COMPLETE
  authorized_by: user, in session, as one Appendix-L engineering-readiness milestone
  is_scientific_milestone_c4: false
  branch: c4-c13-engineering-readiness
  handoff_artifact: reports/handoff/C0_C13_ENGINEERING_HANDOFF.json

  new_canonical_modules:
    - src/prism_fas/search/{plan,coordinate}.py     # the §15.2.2/§15.2.3 bounded engine
    - src/prism_fas/pipeline/portability.py         # backend-neutral batch and paths
    - src/prism_fas/pipeline/handoff.py             # the GPU handoff inventory
    - src/prism_fas/synthesis/gate_profiles.py      # §11.4 STRICT/NOMINAL/PERMISSIVE
    - src/prism_fas/detector/decision_audit.py      # the §13.5 regression guards
    - src/prism_fas/evaluation/source_matrix.py     # the §18 source matrix
    - src/prism_fas/evaluation/source_lock.py       # SOURCE_MATRIX_LOCK_C
  new_adapters: src/prism_fas/pipeline/adapters/c{4,5,6,7,8,9,10,11,12,13}.py
  adapter_design_rule: >-
    unchanged from C0-C3: adapters are THIN. Every model, loss, gate, metric, lock
    and selector is imported from the module that already owns it, and an adapter
    that cannot import its canonical implementation BLOCKS rather than substituting
    one.

  validate_c0_c13:
    command: python train.py --profile validate --from C0 --to C13
    outcome: PASS
    stages: 14
    checks_run: 21
    checks_failed: 0
    provider_calls: 0
  smoke_c0_c13:
    command: python train.py --profile smoke --from C0 --to C13
    outcome: PASS
    stages: 14
    substage_modes_executed: 62
    adapter_checks_run: 242
    adapter_checks_failed: 0
    provider_calls_live: 0
    provider_calls_mock: 0        # the C3 fixture rehearsal re-issued nothing
    scientific_namespace_untouched: true   # 147 files, digest identical before/after
  second_run_idempotency:
    command: python train.py --profile smoke --from C0 --to C13 --resume
    outcome: PASS
    c3_provider_calls: 0          # COMPLETED_VALID requests are never re-issued
    c4_search_trials_reused: 9 of 9
    c7_search_trials_reused: 21 of 21
    c8_rows_skipped_by_identity: 4 of 42
    duplicate_run_ids: 0

  search_plan_identities:
    c4_gpat_coordinate_v1: ab77e964d9c035cf2c3bed209ffac307aebd85c6735879bc3fa3c5efce20d0ec
    c7_detector_coordinate_v1: 62d0022507e732ba89618845fab2c63fec2b7b07f6817b2d541a4f500f459d7b
  c8_source_matrix_identity: a777671fb9142a75369a905f66eee5f0f2ab5c3827f33d3803d52426e2e29af8
  handoff_inventory_identity: 13720f9cd8c670ad401ed07567cdb19ddb3bf448f05e863e431b61f23bc17581

  modal_gpu_seconds_spent: 0
  gemini_calls_this_milestone: 0
  real_target_package_resolved: false
  target_labels_opened: 0

# --- pre-GPU scientific decision closure (this milestone) --------------------
pre_gpu_scientific_decision:
  status: DOSSIER_COMPLETE_DECISION_APPROVED
  is_scientific_execution: false
  branch: pre-gpu-scientific-decision
  branched_from: f8d5a5fab9f253c61399cda5f4031f4b4af0e68c
  dossier: reports/handoff/LR_ANCHOR_DECISION_DOSSIER.json
  dossier_identity: 5997bdcc23927777f2dc66cc48cc51aa7c363a4d8a0cd551137de196f5366ce1
  dossier_markdown: reports/handoff/LR_ANCHOR_DECISION_DOSSIER.md

  learning_rate_decision: APPROVED   # superseded by portable_execution above

  # What the forensic audit established, from code and byte-comparison rather
  # than from config names.
  findings:
    - "No scalar named `learning_rate` exists anywhere. C4 has three per-group LRs
      (encoder 2e-4, recipe 1e-4, generator 2e-4) and C7 has two (backbone 1e-5,
      heads 1e-4). All are consumed by ONE AdamW per component, as disjoint
      parameter groups."
    - "The three C4 groups are all non-empty and together cover 100% of the GPAT
      model's 910,538 trainable parameters. None is historical, inactive, or a
      superset of the others, so no scalar is uniquely applicable."
    - "TRACK G RESOLVES WITHOUT USER INPUT. §13.4.1 forbids Track G from
      instantiating ConvNeXt, PRISMDetector.parameter_groups omits an empty group,
      and backbone_lr therefore controls ZERO parameters under Track G. head_lr is
      the unique inherited anchor — ALREADY_IMPLIED_BY_FROZEN_SPEC, not a choice.
      Version-B M10 row B01 recorded optimizer_groups==['heads'] for the same
      structural reason."
    - "All three LR-bearing configs are byte-identical to Version B, and Version B
      recorded NO learning-rate sweep. Every value is a single inherited setting,
      never a search winner, so nothing elevates one scalar above the others."
    - "The inherited schedulers already treat the LR as a common multiplier over
      grouped anchors: the detector's _lr_lambda returns ONE scalar that LambdaLR
      applies to every group, and GPAT's cosine schedule anchors each group on its
      own base under one shared shape."

  # Trial counts computed by the real SearchPlan, not by hand.
  interpretation_costs:
    c4_gpat:      {D_skip: "3 coords / 9 trials", A_single: "4 / 12", B_multiplier: "4 / 12", C_per_group: "6 / 18"}
    c7_track_r:   {D_skip: "7 coords / 21 trials", A_single: "8 / 24", B_multiplier: "8 / 24", C_per_group: "9 / 27"}
    c7_track_g:   {resolved: "5 coords / 15 trials with head_lr"}
  compliance_classes:
    A_single_scalar: COMPATIBLE_BUT_USER_APPROVAL_REQUIRED
    B_common_multiplier: COMPATIBLE_BUT_USER_APPROVAL_REQUIRED
    C_independent_per_group: SEARCH_ENVELOPE_EXPANSION
    D_skip: NOT_APPLICABLE
    track_g: ALREADY_IMPLIED_BY_FROZEN_SPEC

  recommendation: B_common_multiplier   # for C4 and C7 Track R; RECOMMENDATION_ONLY
  recommendation_implemented: true    # approved and frozen; see portable_execution
  search_plans_unchanged: false      # both identities superseded by the approved decision
  consequence_if_approved: >-
    both frozen plan identities change. c4_gpat_coordinate_v1 ab77e964... and
    c7_detector_coordinate_v1 62d00225... were built with learning_rate AMBIGUOUS
    and skipped; any approved interpretation supersedes them, and the new identity
    is what a later full run must execute against.

  modal_gpu_seconds: 0
  scientific_training_runs: 0
  gemini_calls: 0
  real_target_access: 0
  datasets_opened: 0
  weights_hashed_not_loaded: 5

# --- portable one-command execution closure (this milestone) -----------------
portable_execution:
  status: READY
  branch: portable-one-command-full-run
  commits: [e5fca49, 0ce62bd]     # closure, then the C8 fix the laptop test found
  normal_user_command: "python train.py"
  zero_argument_runner: READY
  dependency_bootstrap: READY
  cpu_rehearsal_mode: READY
  gpu_scientific_auto_mode: READY_NOT_EXECUTED
  compute_profiling: READY_NOT_GPU_VALIDATED
  plot_table_report_generation: READY
  auto_resume: READY
  one_folder_portability: READY

  learning_rate_decision: APPROVED
  lr_c4: B_common_multiplier                # anchor vector 2e-4 : 1e-4 : 2e-4, ratio 2:1:2
  lr_c7_track_r: B_common_multiplier        # anchor vector 1e-5 : 1e-4, ratio 1:10
  lr_c7_track_g: head_lr_unique_applicable_anchor
  lr_decision_config: configs/search/lr_anchor_decision.yaml
  lr_decision_identity: 7ef3492263507d4399828089bbe1af79438bc892e50c8ad732585c1d40c8397c
  lr_decision_record: reports/handoff/LR_ANCHOR_DECISION_RECORD.json
  lr_dossier_preserved: reports/handoff/LR_ANCHOR_DECISION_DOSSIER.json

  frozen_search_plans:
    c4: {identity: 71bfff29bfe1e7ba71d083831a0337a6ae6e0dcfc7f7a75eb9e6f3f3a4ac2b6a, trials: 12}
    c7_track_r: {identity: d671eb002ea3262f2b193ee010db33694cbf031e2bcfbabb73f187c94c46cba3, trials: 24}
    c7_track_g: {identity: e42e91de37621b0d3b18ab910196bf870763a2b34f81bdd1e2c985107a42943b, trials: 12}
  superseded_pre_decision_plans:
    c4: ab77e964d9c035cf2c3bed209ffac307aebd85c6735879bc3fa3c5efce20d0ec
    c7: 62d0022507e732ba89618845fab2c63fec2b7b07f6817b2d541a4f500f459d7b

  new_components:
    - bootstrap.py                                  # stdlib-only; a test asserts it
    - train.py                                      # thin zero-argument entrypoint
    - configs/environment/environment_contract.yaml # python range + declared CUDA profiles
    - configs/execution/rehearsal.yaml              # CPU_FULL_REHEARSAL profile
    - configs/search/lr_anchor_decision.yaml        # the approved LR decision
    - requirements/{base,cpu,cuda-cu129,cuda-cu126,dev,constraints}.txt
    - src/prism_fas/pipeline/runner.py              # intent resolution and preflight
    - src/prism_fas/pipeline/assets.py              # portable asset manifest
    - src/prism_fas/reporting/                      # complexity, resources, history,
                                                    # plots, tables, report, bundle
    - src/prism_fas/search/lr_decision.py           # the approved interpretation

  zero_argument_run:
    command: "python train.py"
    executed_on_laptop: true          # user-authorized test, 2026-08-17
    resolved_intent: CPU_FULL_REHEARSAL
    resolution_reason: "no CUDA GPU matched a declared scientific profile"
    outcome: PASS
    exit_code: 0
    wall_clock_seconds: 94            # 103 s from a cleared rehearsal namespace
    stages: 14
    substage_modes: 56    # CORRECTED from 62 on 2026-08-17. 62 was not reproducible:
                          # all 20 rehearsal runs in MASTER_RUN_INDEX.json record
                          # exactly 56 substage rows, this one included. The smoke
                          # figures elsewhere in this file were NOT re-measured and
                          # are left as recorded.
    provider_calls: 0
    engineering_status_per_stage: SMOKE_PASS
    scientific_status_per_stage: NOT_RUN
    reports_namespace: reports/rehearsal
    runs_namespace: runs/rehearsal
    plots_written: 4
    tables_written: 9
    report_html: reports/rehearsal/final/report.html
    report_is_self_contained: true    # figures embedded as data URIs
    scientific_outputs_modified: 0
    real_target_access: 0
    # Checked independently rather than taken from the runner's own banner: the
    # only paths written were reports/rehearsal/, runs/rehearsal/ and the two
    # state/ cursors. reports/c0..c3, reports/full and assets/recipe_banks were
    # byte-unchanged.
    scientific_namespaces_untouched: true
    reruns_executed: 3
    rerun_output_identical: true      # after the C8 fix below; not before it

  # --- what running it on the laptop actually found -------------------------
  laptop_verification:
    verdict: THREE_DEFECTS_FOUND_AND_FIXED
    found_by: "running `python train.py` three times and diffing the outputs"
    fix_commit: 0ce62bd
    scientific_impact: NONE
    scientific_impact_reason: >
      C8 does not override run_full, so --profile full returns
      BLOCKED(SCIENTIFIC_PATH_NOT_EXERCISED) and the matrix could never have been
      truncated by the rehearsal sampler. No scientific constant, envelope,
      selection rule, quota, seed family or frozen identity was touched.
    defects:
      - id: c8_sample_slid_forward
        severity: HIGH_ENGINEERING
        was: "the rehearsal sampled pending[:SMOKE_ROWS], the first rows still
              PENDING, rather than the first rows of the plan"
        consequence: >
          Resume marks each executed row complete, so the window advanced by two
          on every rerun: different arms each time, and after roughly 21 reruns
          no pending rows would remain, leaving a rehearsal that executes zero
          rows and still reports PASS.
        now: "the sample is the first rows of the plan, in plan order; a sampled
              row that resume already validated is reported from its stored
              artifacts by C8Adapter._reuse_one rather than re-run"
      - id: c8_rollup_last_writer_wins
        severity: MEDIUM_ENGINEERING
        was: "C8_MODEL_COMPLEXITY.json and C8_COMPUTE_RESOURCES.json were written
              inside the per-row loop"
        consequence: "the surviving stage artifact described whichever arm
                      finished last while being labelled as the stage result"
        now: "written once after the loop, holding every executed arm sorted by
              row id; reporting.profiled_entries reads both the single-model
              shape (C4, C7) and the multi-row rollup shape (C8)"
      - id: c8_row_name_collision
        severity: LOW
        was: "rows named detector_{experiment_id}, which repeats across protocols
              and seeds"
        consequence: "two sampled rows rendered as duplicate identical entries"
        now: "named detector_{row_id}"
    also_fixed:
      - id: report_figures_were_linked_not_embedded
        was: "report.html referenced ../plots/*.png, which .gitignore excludes as
              regenerable"
        consequence: "the report rendered four broken images from a fresh clone"
        now: "figures embedded as data URIs; a figure that cannot be read is
              named as missing rather than rendered broken"
        found_by: "opening an actual fresh clone of the pushed branch"
    proof_after_fix:
      consecutive_runs: 3
      complexity_table_byte_identical: true
      sampled_rows: [C-G-RND-P1-s20260806, C-G-RND-P1-s20260807]
      schedule_skipped_stable_at: 2     # was climbing by 2 per run
      rows_reported_reused_on_rerun: true
    regression_tests_added: 5           # tests/pipeline/test_portable_runner.py

  test_suite:
    full_suite: "1820 passed, 7 failed, 101 skipped"
    failures_are_the_inherited_set: true   # reports/c0/C0_TEST_SUITE.json
    new_unexplained_failures: 0
    pipeline_suite_after_c8_fix: "369 passed"

  output_audit_matrix:
    artifact: reports/handoff/OUTPUT_AUDIT_MATRIX.json
    stages_covered: 10        # C4..C13
    missing_writers: 0

  environment:
    manifest: state/ENVIRONMENT_MANIFEST.json
    profile_id: cpu
    profile_status: VALIDATED
    cuda_profiles_status: DECLARED_NOT_VALIDATED_HERE
    supported_python: ">=3.11,<3.14"
    tested_python: "3.13.11"

  pinned_weights_verified: 5   # SigLIP2 (7 files), ConvNeXt, AdaFace, SCRFD, FaceXFormer
  bundle_ready_for_cpu_rehearsal: true
  bundle_ready_for_gpu_science: false   # the derived data trees are absent here

  gpu_seconds: 0
  modal_usage: 0
  gemini_calls: 0
  real_target_access: 0
  scientific_training_runs: 0

# --- final full-path / CUDA-portability / one-folder closure (this milestone) ---
final_full_path_closure:
  status: COMPLETE
  branch: portable-one-command-full-run
  audited_from_head: 77554ee7f1f49121a15d4121320e645610e7c939
  is_scientific_execution: false
  authorized_by: user, in session, as one engineering-closure milestone
  interrupted_and_resumed: >-
    the first attempt was interrupted by a provider 529 exactly while the broad
    regression was running. The regression result was NOT recoverable — no output
    file survived and the suite runs with `-p no:cacheprovider`, so .pytest_cache
    was stale — and it was rerun. Implementation, focused tests, the relocation
    test and the three-run idempotency test were NOT redone; their evidence was
    re-verified intact (idem_1/2/3 byte-identical, schedules differing only in
    generated_at_utc).

  # --- what changed structurally -------------------------------------------
  the_defect_this_milestone_closed: >-
    every C4-C13 adapter had a `run_smoke` and inherited a `run_full` that refused
    with SCIENTIFIC_PATH_NOT_EXERCISED. The rehearsal therefore exercised code the
    scientific run would never reach: the two paths could not drift apart because
    they were never together. They are now ONE `workflow()` per stage,
    parameterized by an ExecutionContext, so rehearsing the path is evidence about
    the scientific path rather than about a parallel one.
  new_canonical_modules:
    - src/prism_fas/pipeline/execution.py        # ExecutionContext: the one place a
                                                 # profile becomes a policy
    - src/prism_fas/pipeline/adapters/sources.py # the single fixture-vs-real seam
    - src/prism_fas/pipeline/gpu_preflight.py    # pre-C4 GPU proof-of-work
    - src/prism_fas/pipeline/preparation.py      # derived-data auto-build
    - src/prism_fas/pipeline/portable_paths.py   # project-relative logical roots
  removed_everywhere: [run_full, run_smoke]

  # --- C4-C13 production FULL path, audited stage by stage ------------------
  c4_to_c13_full_path:
    implemented: true
    placeholder_tokens_remaining: 0   # SCIENTIFIC_PATH_NOT_EXERCISED, NOT_IMPLEMENTED,
                                      # FULL_MODE_PLACEHOLDER, SMOKE_ONLY, TODO
    audit_method: >-
      AST over all ten adapter modules: each defines `workflow`, none defines
      `run_full` or `run_smoke`, none contains a placeholder token. The single
      NotImplementedError in the codebase is the abstract-base guard in
      adapters/common.py, which exists precisely so that there is no placeholder to
      inherit.
    every_stage_blocks_on_data_never_on_code: true
    blocking_inputs_on_this_laptop:
      C4:  [source_packages, gpat_pair_plan, accelerator]
      C5:  [source_packages, gpat_checkpoint_lock, accelerator]
      C6:  [quality_calibration, c5_candidates]
      C7:  [pretrained_weights, c6_matched_banks]
      C8:  [c6_matched_banks, c7_config_lock, source_packages, pretrained_weights,
            accelerator]
      C9:  [c8_runs, c8_acceptance]
      C10: [c9_source_lock, target_feature_package]
      C11: [c9_source_lock, c10_target_lock, target_feature_package, accelerator]
      C12: [c11_lockset, target_labels]
      C13: [c12_statistics]

  # --- C8 FULL matrix contract ---------------------------------------------
  c8_full_contract:
    matrix_identity: a777671fb9142a75369a905f66eee5f0f2ab5c3827f33d3803d52426e2e29af8
    rows_declared: 42
    scientific_rows_scheduled: 42     # ExecutionContext.limit(42, sample=2) == 42
    rehearsal_rows_scheduled: 2       # SAMPLED_TO_BUDGET
    smoke_rows_reachable_from_full: false
    why: >-
      the scientific branch of `limit()` returns the declared count and never reads
      its `sample` argument, so SMOKE_ROWS is not reachable from the scientific path
      even by mistake. A check named
      c8_scientific_cardinality_is_the_complete_matrix asserts it at run time.
    pending_prefix_sampling_affects_full: false
    row_identities_contract_checked: true   # seed family closed, §18.3 replication
                                            # policy, per-row run_identity

  # --- rehearsal / science ancestry isolation ------------------------------
  ancestry_isolation:
    verified: true
    barriers: 4
    barrier_1_namespace: >-
      every inherited input a C4-C13 stage declares resolves under reports/full or
      runs/full. Not one resolves under reports/rehearsal, runs/rehearsal or
      reports/smoke.
    barrier_2_lock_filename: >-
      a rehearsal writes SOURCE_MATRIX_LOCK_C_REHEARSAL.json, not
      SOURCE_MATRIX_LOCK_C.json, so a rehearsal artifact copied into a scientific
      tree would not answer to the name later stages look for.
    barrier_3_eligibility_stamp: >-
      every rehearsal artifact on disk carries scientific_eligible=false and an
      execution_profile of rehearsal or smoke; a test walks the tree and asserts it.
    barrier_4_reporting_scope: >-
      the reporting layer evaluates no rehearsal or smoke string literal; it reads
      only the namespace it is handed, so a full report cannot absorb a rehearsal.
    gpu_execution_must_begin_from: scientific C4 state, never a laptop rehearsal
                                   checkpoint or result

  # --- generic NVIDIA CUDA readiness ---------------------------------------
  cuda_portability:
    hard_coded_to_rtx_5090: false
    model_name_is_a_gate: false      # name is provenance only
    selection_keys: [compute_capability, driver_version]
    categories: [VALIDATED_PROFILE, COMPATIBLE_DECLARED_PROFILE,
                 UNVALIDATED_COMPATIBLE_CANDIDATE, INCOMPATIBLE]
    declared_profiles: [cuda-cu129, cuda-cu126]
    cpu_fallback_for_science: false
    unreported_capability_is_never_graded_upward: true
    classifier_exercised_on: 10 synthetic hosts, no GPU allocated
    classifier_results:
      RTX_5090_cc12.0_drv580:      {profile: cuda-cu129, grade: COMPATIBLE_DECLARED_PROFILE}
      H100_cc9.0_drv560:           {profile: cuda-cu126, grade: COMPATIBLE_DECLARED_PROFILE}
      RTX_4090_cc8.9_drv550:       {profile: cuda-cu126, grade: COMPATIBLE_DECLARED_PROFILE}
      A100_cc8.0_drv535:           {profile: cuda-cu126, grade: COMPATIBLE_DECLARED_PROFILE}
      T4_cc7.5_drv525:             {profile: cuda-cu126, grade: COMPATIBLE_DECLARED_PROFILE}
      RTX_3090_cc8.6_drv550:       {profile: cuda-cu126, grade: COMPATIBLE_DECLARED_PROFILE}
      unlisted_card_cc10.0_drv580: {profile: cuda-cu129, grade: COMPATIBLE_DECLARED_PROFILE}
      RTX_3090_cc8.6_drv470:       {grade: INCOMPATIBLE, why: driver below floor}
      GTX_1080Ti_cc6.1_drv525:     {grade: INCOMPATIBLE, why: capability outside range}
      RTX_5090_capability_unreported: {grade: INCOMPATIBLE, why: never graded upward}
    an_unlisted_card_is_accepted_on_capability: true   # proven by the cc10.0 case
    hardware_validation_status: NOT_YET_EXECUTED       # no real CUDA host here
    gpu_preflight_runs_before_c4: true
    gpu_preflight_probes: [allocate, matmul, build detector, forward, backward,
                           checkpoint save, checkpoint reload, synchronize,
                           memory counters]

  # --- one-folder portability ----------------------------------------------
  one_folder:
    separate_dataset_preparation_command_required: false
    separate_pip_command_required: false
    train_py_arguments_required_for_normal_workflow: 0
    derived_trees_built_automatically: true
    derived_build_steps: [m2_preprocess, m3a_package, m3b_priors, gpat_pairs]
    derived_build_is_resumable: true
    derived_build_delegates_to: [prism_fas.data.m2_runner, prism_fas.data.package,
                                 m3b, prism_fas.synthesis.pair_plan]
    derived_build_blocks_rather_than_fabricates: MISSING_RAW_DATA
    rehearsal_only_reports_what_science_would_build: true   # dry_run, no wasted hours
    # SUPERSEDED by preparation_coverage_closure below. The classification was
    # READY_BY_CONSTRUCTION_NOT_EXERCISED with zero test coverage; the real
    # orchestration is now exercised against fixtures and three defects that were
    # hiding in it are fixed.
    derived_build_evidence_class: ENGINEERING_EXERCISED_WITH_FIXTURES
    derived_build_test_coverage: 50       # tests/pipeline/test_preparation.py
    derived_build_real_path_ever_run: ENGINEERING_ONLY_WITH_STUB_BUILDERS
    derived_build_real_full_data_run: false   # still NOT_RUN; needs the raw corpora
    relocatable_logical_roots:
      - data/raw/casia_fasd
      - data/raw/msu_mfsd
      - data/raw/siw_mv2        # target; not opened before C10
      - weights                 # the pinned model cache
    resolution_order: in_folder wins, else the absolute root in configs/paths.local.yaml
    which_location_was_used_enters_a_scientific_identity: false
    bundle_manifest: PORTABLE_BUNDLE_MANIFEST.json
    bundle_ready_for_full: "YES"     # on this machine; raw roots and weights present
    bundle_blockers: []

  # --- production FULL output audit ----------------------------------------
  production_output_audit:
    writers_are_the_same_code_under_both_contexts: true
    what_the_context_changes: [namespace, cardinality, lock filename, target
                               capability, fixture permission]
    per_stage_summary_naming: "{stage_id}_{PROFILE}.json"   # orchestrator.py:243 —
                                                           # a full run writes
                                                           # reports/full/c4/C4_FULL.json
    covered: [config, run manifest, train history, source and cross-source metrics,
              checkpoint, status, raw evidence, selection evidence, locks,
              parameter counts, FLOPs/MACs, GPU VRAM and timing, throughput,
              inference latency, plots, tables, report.html, MASTER_RUN_INDEX]
    missing_writers: 0
    evidence_class: STRUCTURAL_NOT_OBSERVED
    evidence_class_meaning: >-
      the audit matrix was computed from the artifacts a REHEARSAL produced, because
      C4-C13 have never run scientifically and no reports/full/c4..c13 artifact set
      exists yet. That is evidence about the full path only because the writers are
      the same code under both contexts — which is the property this milestone
      established. It is not an observation of a full run.
    c13_can_aggregate_rehearsal_evidence: false

  # --- exact test evidence, post-fix ---------------------------------------
  tests:
    broad_regression_exact_command: >-
      python -m pytest -q --no-header -p no:cacheprovider
      --continue-on-collection-errors
    broad_regression: {passed: 1877, failed: 7, skipped: 101, seconds: 536.42}
    broad_regression_rerun_reason: interrupted by a 529 and not recoverable
    inherited_failure_set_identical: true    # compared test-id by test-id against
                                             # reports/c0/C0_TEST_SUITE.json
    new_unexplained_failures: 0
    documented_failures_now_passing: 0
    skipped_drift: 0                         # 101 before, 101 after
    previous_baseline: {passed: 1820, failed: 7, skipped: 101}
    net_new_passing_tests: 57
    closure_suite_exact_command: >-
      python -m pytest tests/pipeline/test_full_path_and_ancestry.py -q --no-header
      -p no:cacheprovider
    closure_suite: {passed: 41, failed: 0, skipped: 0}
    pipeline_suite_exact_command: >-
      python -m pytest tests/pipeline -q --no-header -p no:cacheprovider
    pipeline_suite: {passed: 421, failed: 0, skipped: 0}   # as measured AT that
                                                           # milestone; 471 now
    relocation_test: PASS          # completed before the interruption; not redone
    three_run_idempotency: PASS    # idem_1/2/3 byte-identical; schedules differ only
                                   # in generated_at_utc; no counter drift
    tests_weakened_to_obtain_green: 0
    test_style_change: >-
      the closure tests read AST constants rather than matching prose, so a
      docstring may discuss the rehearsal while the assertion still proves no
      rehearsal literal is EVALUATED.

  gpu_seconds: 0
  modal_usage: 0
  gemini_calls: 0
  real_target_access: 0
  scientific_training_runs: 0
  target_labels_opened: 0
  datasets_opened: 0

# --- GPU deployment bootstrap hotfix (this milestone) ------------------------
#
# The first thing the transfer proved is that the one-command promise had never
# been tested on a machine this project did not build. It failed twice on the GPU
# laptop before any science could start, and both failures were real defects
# here, not operator error.
c5_source_pair_plan_freeze:
  status: PARTIAL — the frozen plan is implemented; the render executor is not
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session — both blockers closed as explicit decisions
  base_commit: 876de9bf9a4e4ea7fe2a67bdd6847548c3c9da44

  decisions_closed_by_user:
    - id: C5_CANDIDATE_IDENTITY_VS_QUALITY_CALIBRATION
      ruling: >-
        C5 raw generation identity MUST NOT bind threshold_sha256,
        fingerprint_reference_sha256, calibration_sha256, the selected quality
        profile or any C6 acceptance decision. Those belong to C6 evaluation and
        matched-bank identity. Changing a C6 threshold may change the acceptance
        decision, the provenance, the bank membership and the C6 BANK_LOCK
        identity — never a C5 candidate id or its bytes.
      implemented_as: >-
        a new Version-C module, prism_fas.synthesis.c5_source_pair_plan, whose
        candidate_identity() takes no calibration parameter at all. The
        Version-B SyntheticBankGenerator and candidate_plan are untouched.
    - id: C5_RENDER_TO_LIVE_SAMPLE_MAPPING
      ruling: C5_SOURCE_PAIR_PLAN_V1, transcribed below
      implemented_as: the same module

  frozen_plan:
    name: C5_SOURCE_PAIR_PLAN_V1
    schema_version: prism-c5-source-pair-plan-v1
    plan_seed: 20260806
    live_list: >-
      every label=live row of finalized M3B source_train, sorted by sample_id
      ascending. No source_dev, no target.
    spoof_list: >-
      every label=spoof row of the same manifest, same sort.
    position: p = 8*r + s for recipe ordinal r in 0..255 and slot s in 0..7
    live_assignment: LIVE_LIST[p % len(LIVE_LIST)]
    route_schedule: >-
      s in {0,2,4,6} -> Physics; s in {1,3,5,7} -> GPAT. Exactly 4 and 4 per
      recipe, 1024 and 1024 per arm.
    gpat_domain_schedule: "s in {1,5} -> same_domain; s in {3,7} -> cross_domain"
    gpat_spoof_selection: >-
      key = SHA256(PRISM_C5_SOURCE_PAIR_PLAN_V1 | seed | p | live_id |
      domain_relation); eligible spoof rows sorted by sample_id;
      eligible[int(key[:16],16) % len(eligible)].
    eligibility: >-
      source_train only; source_record_id != the live row's; different subject_id
      whenever both are known; the slot's domain relation enforced. An empty pool
      FAILS CLOSED — no constraint is relaxed and no other render policy is used.
    arm_independence: >-
      the base schedule takes no arm, no recipe bank and no recipe content. The
      p -> live_id map, the route sequence and the GPAT spoof pairing are
      functions of the position alone, so RND, DET and LLM differ only in recipe
      content — which is the treatment under test. This is what keeps a C6
      acceptance-rate difference interpretable.
    cardinality_asserted_on_the_plan: >-
      2048 positions, 1024 Physics + 1024 GPAT, 4+4 per recipe, 2 same-domain +
      2 cross-domain GPAT per recipe, and no GPAT pair sharing a source record.

  identities:
    source_pair_plan_identity: >-
      binds schema, plan name, seed, M3B package identity, ordered live and spoof
      list identities, recipe/slot counts, route schedule, domain schedule,
      eligibility rules, both algorithm names, and a digest over every position's
      (position, route, live, spoof). Excludes nothing that determines it.
    arm_candidate_plan_identity: >-
      the base plan identity as an INPUT, plus arm, that arm's frozen recipe-bank
      identity, the C4 winning GPAT checkpoint SHA, the physics engine version
      and the ontology identity. Three arms therefore differ while provably
      naming the same base schedule.
    candidate_identity: >-
      c5syn_<24 hex> over plan identity, arm, recipe bank, recipe id, ordinal,
      slot, position, route, live target, spoof source or physics-none, package
      identity, ontology identity, generator binding and seed. The generator
      binding is the C4 checkpoint SHA on the GPAT route and the PhysicsEngine
      version on the Physics route. NO calibration field exists in the signature.
      No path, filename, timestamp, subject id or target token.

  # --- built in C5_SCIENTIFIC_RENDER_EXECUTOR --------------------------------
  implemented:
    - C5Adapter.workflow branches on context.is_scientific; the rehearsal path is
      byte-identical and still reaches its fixture batch and untrained generator
    - seven scientific substages: VERIFY_C4_LOCK, LOAD_SOURCE_PAIR_PLAN,
      BUILD_ARM_PLANS, RENDER_CANDIDATES, VERIFY_RAW_CANDIDATES, FINALIZE_C5,
      VERIFY_C5_LOCK
    - c4.verify_gpat_config_lock, extracted to module level and SHARED: C4's own
      VERIFY_LOCK and C5's VERIFY_C4_LOCK run the identical checks
    - src/prism_fas/synthesis/c5_arm_plan.py — the three arm plans over the one
      base schedule, from the three frozen C3 banks (never prism_recipe_bank_m7_v1)
    - src/prism_fas/synthesis/c5_raw_generation.py — the gate-free candidate
      record layer, its reuse decision and its failure retention
    - src/prism_fas/synthesis/c5_render.py — route_bank, build_routes, render_one,
      render_arm, collect_records, completeness; PhysicsRoute / GPATRoute /
      finalize_discrete are imported, never reimplemented
    - CUDA fail-closed for the GPAT route (ScientificDeviceUnavailable)
    - C5_SYNTHESIS_LOCK.json, which reports completion and usability as two facts
    - C6 now requires reports/full/c5/C5_SYNTHESIS_LOCK.json, not the directory
  # --- what is NOT done -----------------------------------------------------
  not_implemented:
    - nothing has been RENDERED. No candidate exists. The branch fails closed at
      VERIFY_C4_LOCK on this laptop because no scientific C4 lock exists here,
      which was verified by driving the real branch under --profile full.
  c5_still_blocks_c6: true
  # --- built in C5_SCIENTIFIC_LOCK_C6_HANDOFF_CLOSURE ------------------------
  lock_closure:
    verifier: prism_fas.pipeline.adapters.c5.verify_c5_synthesis_lock
    shared_by: [C5 VERIFY_C5_LOCK, C6 semantic_preconditions]
    scientific_pass_requires:
      - lock exists, is_scientific_lock, scientific_eligible, not fixture_backed
      - the inherited C4 GPAT lock verifies NOW (shared c4 verifier)
      - every_planned_candidate_is_terminal AND every_planned_candidate_is_usable
      - generated == 6144, failed == 0
      - 2048 verified per arm; 1024 physics and 1024 gpat per arm
      - every candidate's 3 payload files present and re-hashed from BYTES
      - M3B package, source_pair_plan, 3 arm plans, 3 C3 banks, ontology,
        C4 checkpoint SHA and PhysicsEngine version REBUILT NOW all agree
    terminal_but_short: >-
      lock is still written and retained (L.8 forbids winner-only cleanup) but is
      stamped lock_kind=terminal_audit_record, usable_as_c6_input=false. It never
      yields scientific_evidence and never unblocks C6.
    c6_gate: >-
      EngineeringAdapter.semantic_preconditions is a new general hook evaluated
      inside full_precondition_gate. An unsatisfied semantic precondition BLOCKS
      exactly as a missing file does. Presence-only C5 handoff is eliminated.

  # --- NEEDS_SCIENTIFIC_DECISION: C5 transient-failure retry -----------------
  open_decisions:
    - id: C5_TRANSIENT_VS_SEMANTIC_GENERATION_FAILURE
      status: NEEDS_SCIENTIFIC_DECISION
      audited_on: 2026-08-22
      what_the_spec_says: >-
        The rule exists, but it is the RECOVERY LADDER rung L1 in spec section
        0.1, not Appendix L rule L.1 (which is the smoke-vs-science rule).
        Verbatim: "L0 verify/fix implementation defect without changing the
        scientific contract; L1 retry the identical frozen configuration for
        transient/runtime failure; L2 execute the bounded SOURCE_SEARCH envelope
        once; L3 use only a preregistered compatibility fallback explicitly named
        by this spec; L4 STOP and request user approval." Section 0.1 also lists
        "retry of an identical request" under ENGINEERING_ADAPTIVE, permitted
        when scientific identity and output semantics are unchanged.
      current_behaviour: >-
        c5_render.render_arm converts EVERY non-RenderError exception into a
        permanent FAILED_GENERATION record. reuse_decision then returns
        FAILED_GENERATION forever, so a CUDA hiccup or a transient IO error
        permanently costs one candidate out of 2048 — and under the closure above
        that single loss makes the whole arm fail scientific verification.
      why_it_was_not_fixed: >-
        No canonical classifier in this repository can safely separate transient
        from deterministic here. The only classifier that exists,
        prism_fas.llm.contracts.ErrorClass, enumerates PROVIDER failures
        (transport / 5xx / 429 / invalid_candidate vs auth / quota / contract) and
        has no vocabulary for a CUDA, filesystem or codec failure. Writing one
        would be inventing a scientific-retention policy, which the user
        explicitly forbade. Recorded rather than guessed.
      what_is_already_safe: >-
        The one failure class that IS provably deterministic is already distinct:
        RenderError from an empty exact mask is a pure function of the frozen
        inputs and must stay retained under any policy.
      decision_needed_from_user: >-
        whether to add a render-side failure classifier, and if so which failure
        classes are retryable under recovery-ladder L1.
      RESOLVED_BY: C5_RUNTIME_RECOVERY_V1
      resolved_on: 2026-08-22

  # --- FIRST REAL SCIENTIFIC C4/C5 GPU RUN (immutable observation) -----------
  first_scientific_gpu_run:
    c4: {scientific_status: PASS, artifact: reports/full/c4/GPAT_CONFIG_LOCK.json}
    c5:
      planned: 6144
      terminal: 6144
      generated: 6082
      semantic_failed: 62
      runtime_unresolved: 0
      per_arm:
        DET: {generated: 2020, physics: 996, gpat: 1024, physics_failures: 28}
        LLM: {generated: 2034, physics: 1010, gpat: 1024, physics_failures: 14}
        RND: {generated: 2028, physics: 1004, gpat: 1024, physics_failures: 20}
      all_failures: >-
        Physics route, SemanticGenerationFailure, artifact did not survive uint8
        quantization and the exact mask is empty. GPAT: 3072/3072 generated.
      immutability: >-
        the 62 failure records are immutable negative provenance. Never retried,
        deleted, regenerated, re-paired to another live sample or recipe. The
        2048-per-arm budget did not grow and PhysicsEngine/quantization/the
        empty-mask condition are unchanged.
      c6_pre_gate: FEASIBLE   # 996/1010/1004 Physics all clear the 512 floor

  # --- CORRECTED: C5 owns the pool, C6 owns final cardinality ----------------
  c5_stage_ownership_correction:
    defect: >-
      the implementation required generated == 6144 and failed == 0 before C5
      could claim completion or unblock C6. That is STRONGER than the frozen
      contract and would have rejected the real 6082+62 run.
    spec_evidence:
      - "§10.4: scientific synthesis budget is fixed at 2048 candidate RENDERS per
        arm = 256 recipes x 8, with exactly 4 Physics and 4 GPAT per recipe
        BEFORE the common quality gate. Final accepted bank is exactly 1024/arm."
      - "C5 hard acceptance (stage table): same base live list / route budget; no
        target. Nothing about zero failures."
      - "§11.3: Mỗi arm bắt đầu từ 2048 candidate renders. Nếu một arm không đạt
        1024 dưới frozen render budget/gate, C6 FAILS thay vì nới gate riêng cho
        arm đó."
      - "§11.4: C6 selects the strictest profile that yields >=1024 accepted in
        EVERY arm with exactly 512 Physics + 512 GPAT feasible. If none
        qualifies, C6 FAILS."
    c5_scientific_complete_requires:
      - 6144 planned positions, 6144 terminal, 0 runtime-unresolved
      - 2048 terminal slots per arm over the frozen 1024/1024 PLANNED route split
      - every GENERATED record: current identity, 3 payload files, bytes re-hashed
      - every FAILED_GENERATION record: current identity, deterministic
        candidate-semantic, error_type SemanticGenerationFailure,
        replacement_generated false, and NO declared payloads
      - generated + semantic_failed == planned
      - C4 / M3B / C3 / source-pair / ontology / PhysicsEngine identities rebuild now
      - source_dev unopened, target unopened
    c5_scientific_complete_does_not_require: [failed == 0, generated == 6144]
    c6_pre_gate_input_ready_requires: >-
      additionally >=512 generated Physics and >=512 generated GPAT in EVERY arm.
      Arithmetic only; no threshold is applied and nothing is accepted or
      rejected. Below the floor C6 is impossible before the gate runs.
    lock_schema: >-
      the lock represents a COMPLETE FROZEN CANDIDATE POOL. planned / terminal /
      generated / semantic_failed / runtime_unresolved are kept apart, plus
      usable_generated_by_arm_and_route and c6_pre_gate_route_floor_feasible.
      `every_planned_candidate_is_usable` survives as DESCRIPTIVE ONLY and is no
      longer any stage's acceptance predicate.
    old_lock_preservation: >-
      _archive_superseded_lock copies an existing lock byte-for-byte to
      reports/full/c5/superseded/C5_SYNTHESIS_LOCK_<sha16>.json, binds its
      SHA-256 into the replacement under `supersedes`, and records
      supersedes_verifier_semantics plus the reason. No CANDIDATE.json and no
      payload is touched.

  # --- C6 AUDIT (Phase B) ----------------------------------------------------
  c6_audit:
    defect_c6_1_self_dependency:
      status: FIXED
      was: >-
        required_inputs demanded reports/full/c6/QUALITY_CALIBRATION.json before
        C6 could start, but §11.4 fits NOMINAL from the source_train benign
        population AT C6. It is a C6 OUTPUT, so C6 depended on itself and the
        only way to satisfy it would have been a hand-written fitted threshold.
      fix: the RequiredInput was removed; no calibration file was fabricated.
    defect_c6_2_full_uses_engineering_workflow:
      status: CONFIRMED_NOT_FIXED
      why: fixing it means writing the scientific executor, which is blocked below.
      containment: >-
        C6 has no _scientific_workflow, claims no scientific_evidence anywhere,
        and under `full` reaches the shared precondition gate and BLOCKS. The
        engineering path is unchanged.
    determined_by_frozen_contract_or_canonical_modules:
      - source_train reference population: quality_calibration.calibrate
      - metric backends/weights: QualityBackends, quality_models
      - inherited Version-B thresholds: gate_profiles NOMINAL inheritance
      - percentile derivation without an inherited threshold: §11.4 gives
        10th/5th/1st (higher-is-better) and 90th/95th/99th (lower-is-better)
      - quantile method: np.percentile as already used by quality_calibration
      - STRICT/NOMINAL/PERMISSIVE formulas: gate_profiles.derive_profile
      - range-safe constraints: gate_profiles.RANGE_SAFE
      - candidate evaluation: synthetic_bank.CandidateEvaluator, quality_gate.evaluate
      - shortcut/reliability gates: prism_fas.evaluation.reliability (§17.3)
      - profile selection: gate_profiles.select_profile, conjunctive over arms
      - BANK_LOCK contents: synthetic_export / existing BANK_LOCK contract
      - resume/idempotency: the shared identity-aware framework

  open_decisions_c6:
    - id: C6_MATCHED_BANK_SELECTOR
      status: NEEDS_SCIENTIFIC_DECISION
      audited_on: 2026-08-22
      what_the_spec_says: >-
        §11.3: "1024 accepted samples/arm, gồm 512 Physics + 512 GPAT, selected
        deterministically để cân bằng source domain, route, recipe coverage và
        base live IDs." The route split is exact. The other three balancing
        dimensions are NAMED and no algorithm is given.
      what_is_missing:
        - no priority order among source domain, recipe coverage and base live ID
        - no target distribution (proportional to the accepted pool? uniform?
          proportional to the source package?)
        - no algorithm (greedy round-robin? stratified quota? largest remainder?)
        - no tie-break when a stratum holds more accepted candidates than its quota
        - no redistribution rule when a stratum holds fewer than its quota
      why_it_cannot_be_inferred: >-
        gate_profiles.matched_bank_plan returns COUNTS and the sentence "balanced
        deterministically over source domain, route, recipe coverage and base
        live IDs (§11.3)". It never returns which candidates. No other module
        selects a subset of accepted candidates, and Version B has none because
        Version B is the confounded design this rule exists to remove.
      result_affecting: >-
        yes. Which 512 Physics candidates enter each arm's bank changes C7
        training and every downstream number, so a guessed tie-break would be an
        unrecorded scientific choice.
      decision_needed_from_user: >-
        the exact deterministic selector: dimension priority, target
        distribution, fill algorithm, tie-break and deficit redistribution.
      RESOLVED_BY: C6_MATCHED_BANK_SELECTOR_V1
      resolved_on: 2026-08-23

  # --- OPEN: C6 quality-backend execution device -----------------------------
  open_decisions_c6_device:
    - id: C6_QUALITY_BACKEND_DEVICE
      status: NEEDS_SCIENTIFIC_DECISION
      audited_on: 2026-08-23
      why_it_matters: >-
        QualityBackends sends the SCRFD provider, FaceXFormer parsing and the
        AdaFace embedding to one device. Every tau is a percentile (p1/p99) over
        a population of those measurements, so CPU and CUDA kernels can move a
        threshold. The choice is result-affecting.
      evidence_searched:
        - "v1.5 spec: the word 'device' occurs 0 times. The precision/backend
          clauses govern TRAINING runs (GPU model, precision mode, microbatch,
          effective batch size), not the quality-metric backends."
        - "configs/synthesis/quality_gate_m8.yaml: no device or provider field."
        - "QualityBackends(weight_root, *, device='cpu'): a Python signature
          default, not a declared scientific contract."
        - "both existing call sites (cli.main, structural_calibration) take the
          device from their caller; the CLI exposes it as an operator flag."
        - "quality_calibration.calibrate records `device` as run PROVENANCE in
          its output, which is how a recorded input behaves, not a frozen one."
        - "frozen Version-B reports/m8/quality_calibration.json recorded
          device='cuda' on an NVIDIA L4 (torch 2.5.1+cu121, cuDNN 90100). The
          Version-C host is an RTX 5090, so 'inherit cuda' would not reproduce
          those numbers either — 'cuda' is a family, not a device."
        - "c4._scientific_device and c5_render.scientific_device require CUDA,
          but each is justified by a training/rendering precision contract and
          neither claims to govern measurement backends. Extending them here
          would be stretching a policy past its stated scope."
        - "gpat_trainer.resolve_device is availability-based ('cuda if
          available'), an operational helper rather than a scientific policy."
      not_chosen_by: >-
        runtime speed, RTX 5090 availability, the observed C5 outcome, which
        option makes C6 pass, or any downstream result. No device was picked.
      RESOLVED_BY: user decision, 2026-08-23 — C6_QUALITY_BACKEND_DEVICE = "cuda"
      resolution: >-
        FROZEN by explicit user decision, preregistered BEFORE any C6 fitted
        threshold, candidate quality result, acceptance count, profile
        feasibility result or matched-bank result was observed, with
        target_access = 0. "cuda" is the DEVICE FAMILY / execution contract; the
        GPU model, driver, CUDA runtime and PyTorch/ORT versions are RUN
        PROVENANCE and are recorded per run. No bitwise reproduction across
        NVIDIA models is claimed (Version B measured on an L4; the Version-C host
        is an RTX 5090). No availability-based fallback: if CUDA is absent C6
        BLOCKS with C6_QUALITY_BACKEND_DEVICE_UNAVAILABLE and never drops to CPU.
      superseded_behaviour: >-
        FROZEN_QUALITY_BACKEND_DEVICE was None, so C6 BLOCKED at
        FIT_NOMINAL_CALIBRATION with reason_code
        C6_QUALITY_BACKEND_DEVICE_NEEDS_SCIENTIFIC_DECISION and the audit list
        above. Setting that constant is the whole of the implementation once the
        decision is made.
      related_observation: >-
        §11.4 says NOMINAL uses the unique inherited Version-B threshold when
        semantically compatible, and only derives percentiles for metrics with no
        compatible inherited threshold. The C6 executor currently fits every
        threshold fresh. Whether C6 should inherit the six frozen Version-B taus
        instead is a SEPARATE question, not touched by this hotfix, and it would
        change how much the device choice affects.

  # --- RECONCILED: §11.4 NOMINAL threshold inheritance -----------------------
  c6_threshold_inheritance:
    status: RECONCILED
    decided_on: 2026-08-23
    rule: >-
      §11.4: for each gate metric NOMINAL uses the unique inherited Version-B
      threshold when semantically compatible; only a metric with no compatible
      inherited threshold is derived from the frozen source reference
      distribution. The first C6 executor fitted every NOMINAL fresh, which takes
      the second branch unconditionally.
    defect_it_prevented: >-
      quality_calibration.calibrate recomputes tau_id as p1 benign
      self-similarity (~0.9995), tau_lm as p99 benign (~0.00214) and tau_parse as
      p1 benign (~0.87478). Those are precisely the v1 values Version B itself
      examined and replaced, so refitting would have resurrected superseded
      science.
    version_b_supersession_chain:
      v1: reports/m8/quality_calibration.json — the M8 base calibration
      v2: >-
        reports/m8/quality_calibration_v2.json — tau_id_v2 = max(tau_genuine,
        tau_impostor); demotes v1's value to v1_tau_id_informational_only
      v3: >-
        reports/m8/quality_calibration_v3.json — tau_lm_v3, tau_parse_v3, with
        tau_lm_v1_superseded and tau_parse_v1_superseded recorded, carrying the
        rest forward as unchanged_from_v2
    authoritative_artifact:
      path: reports/m8/quality_calibration_v3.json
      sha256: a21cb3e168ab04b1f1fc06b4cc311a12357316e68d2cfcdc6f82395aa08d4c2c
      threshold_sha256: 8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19
      version_b_commit: 7799f7decd35db6987ce4578824e5bd8d9eab4ae
      unique_final_value_per_metric: true   # no metric required a choice
    inherited_nominal:
      tau_fd: 0.5                    # pinned SCRFD production threshold
      tau_id: 0.547440037939055      # v2
      tau_lm: 0.00836817528937794    # v3
      tau_parse: 0.7094826178704915  # v3
      tau_out: 0.0                   # FROZEN_RANGE_CONSTRAINT, never profiled
      tau_fp: 5.687657785453908
    compatibility_basis: >-
      provable identity rather than name matching. quality_gate.py,
      quality_calibration.py, quality_models.py, synthetic_bank.py,
      identity_calibration.py, structural_calibration.py and fingerprint.py are
      byte-identical between the frozen Version-B tree and Version C; the three
      pinned models resolve to the same SHA-256s; and Version B calibrated on the
      same M3B package (b1cf29b6…dc6). Same measurement code, same models, same
      population, same comparators, same scale.
    source_reference_derived: []      # every metric has a compatible inherited value
    cross_check: >-
      the assembled NOMINAL identity equals Version-B's own recorded
      threshold_sha256, so the inheritance is exact rather than approximate.
    note_on_derived_branch: >-
      the source-reference percentile branch stays implemented and tested; a
      future metric without an inherited threshold must take it. calibrate() is
      still run, because it supplies the fingerprint reference and the population
      evidence the evaluator needs — it simply no longer decides the thresholds.

  # --- FIXED: C6 final threshold identity ------------------------------------
  c6_final_threshold_identity:
    status: FIXED
    fixed_on: 2026-08-23
    defect: >-
      the §11.4 assembly replaced payload["thresholds"] with the inherited
      NOMINAL but left payload["threshold_sha256"] as the calibrator's hash of
      the map it had just superseded. FrozenCalibration.load recomputes the hash
      from the thresholds it is about to hand the evaluator, found the
      disagreement and refused. The consumer was right; the producer was wrong.
    where_the_gpu_run_stopped: >-
      _evaluate_generated_candidates, at FrozenCalibration.load. CUDA backend
      construction and source_train calibration had succeeded and
      QUALITY_CALIBRATION.json was written; NO candidate was measured, no
      acceptance count existed, no profile outcome existed, no matched bank was
      selected, target_access stayed 0.
    two_identities_now_kept_apart:
      final_scientific_pair:
        thresholds: the assembled §11.4 NOMINAL
        threshold_sha256: 8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19
        equals_nominal_identity_sha256: true
        equals_version_b_threshold_sha256: true   # under the all-inherited contract
        consumed_by: CandidateEvaluator via FrozenCalibration
      calibrator_fitted_pair:
        calibrator_fitted_thresholds: the source-reference fitted values
        calibrator_fitted_threshold_sha256: 4798a392243c85f89b37a14dc5195863…
        role: provenance and population evidence only; never the gate
    implementation: >-
      _final_calibration_payload builds ONE canonical payload used for both
      state["calibration"] and QUALITY_CALIBRATION.json, replaces the thresholds
      and their hash together, preserves the fitted pair under its own names, and
      self-checks the consumer's invariant before writing. FrozenCalibration.load
      was NOT weakened, special-cased or given a fallback field.
    threshold_values_changed: none
    downstream_identity_audit: >-
      every profile, bank-lock and selector identity derives from
      threshold_identity(profile.thresholds), which hashes the profile's own
      values. No downstream consumer ever bound the calibrator-fitted SHA, so no
      field was renamed.
    superseded_note: >-
      an earlier revision recorded the 12-decimal profile rounding as expected
      behaviour. It was corrected in C6_PROFILE_NUMERIC_IDENTITY below: the
      rounding is gone and the NOMINAL profile now equals the inherited values
      exactly, so the profile values identity, the calibration threshold_sha256
      and Version-B's threshold hash all coincide.
    profile_source_label: >-
      "source_train NOMINAL fitted at C6" became inaccurate after inheritance and
      is now NOMINAL_SOURCE_LABEL. It is provenance metadata only —
      threshold_identity hashes the threshold VALUES — and a test proves two
      different labels yield identical profile identities.
    invalid_gpu_artifact: >-
      archive_superseded_calibration copies any existing QUALITY_CALIBRATION.json
      byte-for-byte to reports/full/c6/superseded/, binds its SHA-256 and reason
      into the replacement, and records archived_was_self_consistent=false and
      is_scientific_lock=false. The invalid artifact is preserved as diagnostic
      evidence and never treated as a lock. It was not hand-edited.
    onnxruntime_warnings: >-
      the GPU log emitted repeated SCRFD "Expected shape … does not match actual
      shape" warnings. They did NOT cause this failure — calibration completed and
      execution advanced past it. Recorded as an operational observation; SCRFD
      input size, provider and model are unchanged. A separate audit is warranted
      only if detector outputs later prove malformed.

  # --- FIXED: C6 §11.4 profile numeric identity -------------------------------
  c6_profile_numeric_identity:
    status: FIXED
    fixed_on: 2026-08-23
    defect: >-
      gate_profiles.derive_profile ended with round(value, 12) + 0.0, applied to
      NOMINAL, STRICT and PERMISSIVE alike. §11.4 specifies four formulas and a
      range-safe exemption and no rounding, and for an inherited metric NOMINAL
      must be the Version-B threshold EXACTLY. The quantization turned tau_id
      0.547440037939055 into 0.547440037939, so the NOMINAL profile no longer
      contained the value it is required to inherit.
    origin: >-
      commit f8d5a5f (2026-08-17), the C0-C13 engineering-readiness milestone.
      It predates every scientific executor and the §11.4 inheritance
      reconciliation. NOMINAL was fixture-derived then, so the rounding was
      cosmetic; it became result-affecting only once NOMINAL became an exact
      inherited value.
    fix: >-
      return {name: float(value) + 0.0 ...}. No replacement rounding, Decimal
      quantization, string formatting, epsilon or nextafter. The trailing
      "+ 0.0" is retained solely to canonicalize -0.0; it is exact for every
      other float and a test proves it changes no inherited value bit-for-bit.
    numeric_contract:
      NOMINAL: exact assembled value, no quantization
      higher_is_better_STRICT: a + 0.10 * (1.0 - a)
      higher_is_better_PERMISSIVE: max(0.0, 0.90 * a)
      lower_is_better_STRICT: 0.90 * a
      lower_is_better_PERMISSIVE: 1.10 * a
      RANGE_SAFE: exact original value, never profiled
    identity_consequences:
      nominal_profile_equals_inherited: true      # value for value
      threshold_values_identity: >-
        threshold_identity(NOMINAL profile) == Thresholds.from_dict(INHERITED).sha256()
        == VERSION_B_THRESHOLD_SHA256 == the calibration artifact threshold_sha256.
        Four names for one hash of one set of values, so agreement is legitimate
        rather than forced.
      gate_profile_identity: >-
        GateProfile.identity additionally binds the profile NAME, so it is a hash
        of a structurally different object and remains different from the
        threshold-values identity. Not forced equal; STRICT, NOMINAL and
        PERMISSIVE keep three distinct GateProfile identities.
      strict_permissive_identities_changed: >-
        yes, relative to the pre-fix implementation. No valid scientific C6
        evidence is invalidated: no candidate was ever evaluated, no acceptance
        count observed, no profile selected and no matched bank built.
    superseded_artifacts: >-
      archive_superseded_artifact copies any pre-fix C6_GATE_PROFILES.json
      byte-for-byte to reports/full/c6/superseded/, binds its SHA and reason, and
      marks it is_scientific_lock=false. A pre-fix profiles artifact is
      diagnostic evidence and is not valid scientific input for the corrected
      executor.

  # --- FIXED: C6 raw metric envelope ------------------------------------------
  c6_metric_envelope:
    status: FIXED
    fixed_on: 2026-08-23
    defect: >-
      CandidateEvaluator.evaluate returns quality_gate.evaluate(metrics,
      calibration.thresholds) — a gate ENVELOPE with the raw measurements nested
      under ["metrics"] and an acceptance decision already taken under the
      calibration's NOMINAL. evaluate_pool stored the envelope; gate_candidates
      then passed it to quality_gate.evaluate as if it were the flat metric map,
      so the run died with "metric 'face_detection_score' is missing".
    fix_location: >-
      the C6 adapter boundary only. CandidateEvaluator is inherited canonical
      measurement code and is byte-identical to Version B's, which is the basis
      of the §11.4 compatibility argument, so it was NOT modified.
      c6_scientific.raw_metrics_of unwraps ["metrics"], validates the nine
      required fields plus the two canonical diagnostics, refuses a non-finite
      value or an unrecognized field, and drops every threshold-dependent field.
    contract: >-
      measure once, gate three times. The stored measurement is
      threshold-independent; STRICT/NOMINAL/PERMISSIVE are applied afterwards by
      gate_candidates. The evaluator's embedded NOMINAL verdict is discarded —
      reusing it would give NOMINAL a privileged position among three profiles
      that must be assessed identically.
    gpu_state_at_failure:
      candidates_measured: yes, EVALUATE_GENERATED_CANDIDATES completed
      valid_profile_outcome: none — the crash was inside
        CHECK_PROFILE_MATCHED_FEASIBILITY before any profile was assessed
      valid_matched_bank: none
      embedded_nominal_counts: >-
        computed as a side-effect inside each evaluator call. NOT inspected, NOT
        used to tune any threshold or selector, and NOT scientific selection
        evidence.
      target_access: 0

  # --- FIXED: C6 reliability fail-open ----------------------------------------
  c6_reliability_fail_open:
    status: ELIMINATED
    fixed_on: 2026-08-23
    defect: >-
      _run_reliability_gates set state["reliability"] = {} and reported success;
      the consumer read `all(...) if state["reliability"] else True`, so ZERO
      executed gates was interpreted as every gate passing. §11.4 requires the
      selected profile to PASS the mandatory gates, and "none were run" is not a
      pass.
    fix: >-
      the substage now BLOCKS with reason code
      C6_RELIABILITY_SEQUENCE_NEEDS_SCIENTIFIC_DECISION when no gate produced a
      verdict or the required set is unresolved; the consumer computes
      `bool(reliability) and all(reliability.values())`; and
      c6_scientific.assess_profile no longer defaults reliability_passed to True
      — the argument is required.

  # --- FROZEN: C6_RELIABILITY_SEQUENCE = OPTION B -----------------------------
  c6_reliability_sequence:
    status: FROZEN
    resolved_by: explicit user scientific decision
    option: OPTION_B_POST_SELECTION_CLOSURE_GATE
    constant: c6.C6_RELIABILITY_SEQUENCE = "OPTION_B_POST_SELECTION_CLOSURE_GATE"
    frozen_on: 2026-08-23
    not_observed_to_choose_it: >-
      BA_sep was NOT observed. The previous GPU run carried embedded NOMINAL gate
      side-effects from CandidateEvaluator but produced no valid profile or
      matched-bank outcome, and those side-effects were not inspected to make
      this decision. target_access = 0 throughout.
    sequence:
      - VERIFY_C5_POOL
      - BUILD_SOURCE_REFERENCE
      - FIT_NOMINAL_CALIBRATION
      - BUILD_COMMON_PROFILES
      - EVALUATE_GENERATED_CANDIDATES
      - CHECK_PROFILE_MATCHED_FEASIBILITY
      - SELECT_STRICTEST_PROFILE          # profile frozen here, lock written
      - BUILD_MATCHED_BANKS               # final banks exist here
      - RUN_BANK_LEVEL_RELIABILITY        # closure gate runs here
      - VERIFY_C6_LOCKS                   # closes only on PASSED
    selection_inputs: >-
      STRICT -> NOMINAL -> PERMISSIVE order, per-arm Physics/GPAT floor, common
      source-domain matched feasibility, C6_MATCHED_BANK_SELECTOR_V1. NO
      reliability input: MatchedFeasibility has no reliability field and
      assess_profile has no reliability parameter, so it cannot leak in.
    pass_rule: >-
      every arm must satisfy BA_sep_arm <= 0.75; equivalently
      max(BA_sep_RND, BA_sep_DET, BA_sep_LLM) <= 0.75. This is the C6 hard gate —
      no primary synthetic arm is trivially identifiable — and is NOT the C-H4
      SUPPORT rule, which additionally asks whether LLM beats DET and RND and
      adds validity and diversity conditions.
    failure_semantics: >-
      C6 FAILS. The selected profile stays frozen and is never reopened: no
      looser profile, no stricter profile, no changed candidate selection, no
      discarded arm, no reseeded probe, no widened ceiling. Selected profile,
      bank artifacts and probe outputs are preserved as negative provenance and
      no usable downstream C6 PASS lock is written.
    tri_state: >-
      NOT_YET_APPLICABLE before the final banks exist; then PASSED / FAILED /
      BLOCKED. C6 closes only on PASSED, and NOT_YET_APPLICABLE is never a pass.
    detector_reliability_staging:
      stage: C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C
      rule: >-
        detector-dependent tests are not C6 selection inputs and are not required
        to build the C6 bank-level lock, because no detector exists at C6. They
        execute after C8 source-only detector training and must be resolved
        before SOURCE_MATRIX_LOCK_C closes at C9. No reliability decision may be
        newly tuned from target information at C10/C11. Residual sensitivity is
        NOT moved into C6.
      criteria: >-
        only the stage/deadline is frozen here. Individual detector-level
        pass/fail criteria remain typed future NEEDS_SCIENTIFIC_DECISION items to
        be settled before their first execution.
    classification:
      bank_level: [synthetic_vs_real_spoof_probe]
      detector_level: [residual_scale_zero, recipe_region_shift, artifact_map_swap,
                       cross_route_synthetic, benign_jpeg_corruption,
                       benign_resize_corruption, benign_color_corruption,
                       crop_padding_interpolation]
      no_legitimate_population: [benign_glasses_makeup_lowlight]   # stays BLOCKED

  # --- FIXED: C6 provenance closure accounting --------------------------------
  c6_provenance_closure:
    status: FIXED
    fixed_on: 2026-08-23
    defect: >-
      _build_matched_banks passed [] where provenance_closure expects
      semantic_failure_ids. candidate_pool() is built from the frozen PLAN, so
      its keys include every planned slot including the ones that ended in a
      terminal SemanticGenerationFailure. Those slots have no payload, are never
      measured and carry no gate decision, so with an empty failure set they were
      in `planned` but never in `covered` and closed was False. All three bank
      locks failed for that reason.
    confirmed_before_editing: >-
      reproduced with a production-shaped fixture: semantic_failure_ids=[] gives
      closed=False with the failed id in `unaccounted`; the actual ids give
      closed=True. No scientific selection was altered to make it pass.
    fix: >-
      verify_c5_candidates now returns semantic_failed_candidate_ids and
      semantic_failed_candidate_ids_by_arm, re-derived while it already validates
      every planned slot — so the ids are the ones the strict verifier itself
      accepted, never copied from a lock and never inferred from a missing gate
      decision. _verify_c5_pool carries them in state and _build_matched_banks
      passes the per-arm set.
    lock_consistency_check: >-
      the C5 lock's historical failed_candidate_ids are compared as SETS against
      the re-derived set, because the lock truncates its own list and its order
      is a representation detail. Derived may legitimately exceed locked; the
      reverse is a real disagreement.
    partition_contract: >-
      the frozen planned schedule must partition EXACTLY into selected +
      accepted_not_selected + rejected + semantic_failed, pairwise disjoint,
      unioning to all planned ids, with selected subset of accepted and no
      semantic failure carrying a gate decision. Overlaps and decision-carrying
      failures are now reported and refuse closure.
    result_invariance: >-
      provenance accounting is a pure reader. Profile thresholds, profile order,
      the selection rule, candidate metrics, gate decisions, q, selector V1,
      source-domain quotas, exposure priorities, selected candidate ids,
      selected_set_sha256, selector identity and the C5 bytes are all unchanged.
      A production-path test re-runs the selector and asserts identical selected
      sets and quotas.
    strict_verifier: unchanged. The producer was made truthful, exactly as with
      the earlier FrozenCalibration hash defect.

  # --- ENGINEERING NEGATIVE EVIDENCE: the C6 lock-verification run ------------
  c6_lock_verification_run_2026_08_23:
    stage: C6
    scientific_progress:
      profile_assessment: VALID
      profile_selection: VALID
      matched_bank_construction: VALID
    final_status: BLOCKED at VERIFY_C6_LOCKS
    failed_checks: [c6_bank_lock_rnd_verifies, c6_bank_lock_det_verifies,
                    c6_bank_lock_llm_verifies]
    category: ENGINEERING_PROVENANCE_ACCOUNTING_FAILURE
    scientific_negative_result: false
    paper_use: reproducibility / provenance / fail-closed methodology evidence
    note: >-
      the selected profile and the matched banks are NOT a scientific failure.
      The failed bank-lock artifacts are retained; a corrected run supersedes
      them under the existing archive policy with SHA-256 and reason.
    target_access: 0
    onnx_warnings: >-
      the SCRFD VerifyOutputSizes warnings did not cause this failure; detector
      model, input size and provider unchanged.

  # --- FIXED (reporting only): the stale C3 blocker line ----------------------
  c3_blocker_line_reporting_fix:
    status: FIXED_REPORTING_ONLY
    finding: >-
      the "C3 live scientific generation remains gated" line was appended for
      every full-profile invocation and never consulted the requested stage
      range, so a scoped --from C5 --to C6 run printed it too.
    execution_impact: >-
      none. `blockers` becomes `notes` on the state record and a report field;
      nothing in the orchestrator branches on it. Verified by grep for
      `if blockers` / `not blockers` / `len(blockers)` — no such branch exists.
    fix: the line is emitted only when C3 is actually in the selected range.
    not_touched: >-
      no C3 artifact, lock, authorization flag or scientific evidence was
      changed, and C3 was not reopened.

  # --- FIXED: C6 gate-profile threshold type contract -------------------------
  c6_threshold_type_contract:
    status: FIXED
    fixed_on: 2026-08-23
    defect: >-
      GateProfile.thresholds is a dict[str, float] and quality_gate.evaluate
      reads thresholds.tau_fd, so the gate needs a quality_gate.Thresholds. The
      scientific adapter passed the raw mapping into gate_candidates, and the run
      crashed at the first gating call with "'dict' object has no attribute
      'tau_fd'".
    category: producer/consumer TYPE-CONTRACT mismatch. Not a threshold value,
      §11.4, calibration, CUDA, measurement or matched-bank problem.
    fix: >-
      one call site: profile.as_thresholds(), which is GateProfile's own
      conversion and what the engineering rehearsal already used. No threshold
      value changed; derive_profile, Thresholds, quality_gate.evaluate,
      CandidateEvaluator, raw metric extraction and selector V1 untouched.
    type_contract_tightened: >-
      gate_candidates' `thresholds: Any` became `thresholds: Thresholds` with a
      fail-fast isinstance refusal naming as_thresholds(). `Any` is how the dict
      reached evaluate silently. No dict-accepting compatibility path was added:
      two accepted representations at the gating boundary is what allowed the
      mismatch.
    call_site_audit:
      gate_execution_needs_Thresholds:
        - c6.py _check_profile_matched_feasibility   # FIXED
        - c6.py engineering _apply_common_gate       # already correct
      mapping_is_correct_and_unchanged:
        - c6.py threshold_identity(profile.thresholds)      # hashing
        - c6.py artifact "thresholds": dict(profile.thresholds)  # serialization
        - c6.py engineering unprofiled value comparison
        - gate_profiles.py matched_bank_plan payload        # serialization
    why_tests_missed_it: >-
      every existing test built the object itself —
      Thresholds.from_dict(profiles[name].thresholds) — performing the conversion
      the production code omitted, and the engineering path called
      as_thresholds(). Neither exercised the scientific adapter's wiring. The new
      regression drives the real _check_profile_matched_feasibility and
      intercepts quality_gate.evaluate to assert a real Thresholds arrives.

  # --- ENGINEERING NEGATIVE EVIDENCE: the failed GPU run ----------------------
  c6_failed_run_2026_08_23:
    stage: C6
    substage: CHECK_PROFILE_MATCHED_FEASIBILITY
    category: IMPLEMENTATION_CONTRACT_FAILURE
    reason: GateProfile threshold dict passed where a Thresholds object is required
    candidate_measurement_completed: true
    raw_metric_envelope_correction: worked
    profile_assessment_completed: false
    scientific_result_observed: false
    acceptance_counts_for_any_profile: none
    profile_selected: none
    matched_bank: none
    target_access: 0
    embedded_nominal_side_effect: >-
      still discarded, and NOT inspected to choose or validate this fix.
    paper_use: >-
      reproducibility / fail-closed implementation evidence. NOT a primary
      scientific result and never mixed with an actual scientific gate failure.
    retention: the log and diagnostics are kept, not deleted.
    onnx_warnings: >-
      the repeated SCRFD VerifyOutputSizes warnings did not cause this traceback.
      SCRFD model, input size and provider are unchanged.

  # --- SUPERSEDING DECISION: BA_sep moves to the detector stage ---------------
  synthetic_vs_real_reliability_stage:
    status: FROZEN
    constant: SYNTHETIC_VS_REAL_RELIABILITY_STAGE
    value: C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C
    resolved_by: explicit user scientific decision
    frozen_on: 2026-08-23
    supersedes: >-
      OPTION_B_POST_SELECTION_CLOSURE_GATE, with respect to the PLACEMENT of
      BA_sep only. Option B's ordering guarantee is kept — reliability is not a
      profile-selection input — and C6 simply has no reliability substage.
    c6_ba_sep_probe_protocol: SUPERSEDED_BY_DETECTOR_LEVEL_STAGING
    c6_bank_level_ba_probe: NOT_APPLICABLE_AT_C6
    deferral_reason: DEFERRED_BY_FROZEN_PROTOCOL_DECISION

    # The frozen text does not compose. Recorded, not silently rewritten.
    staging_incompatibility_in_v1_5:
      - "§3.1.1 evaluates BA_sep AFTER the common C6 synthetic gate is frozen"
      - "§17 places the reliability gates BEFORE P3 target evaluation"
      - "the C6 stage row reads 'shortcut gates pass or STOP'"
      - "the only canonical synthetic-vs-real probe uses DETECTOR evidence
        (p_global, s_region, nine normalized regional distances)"
      - "C6 has no detector; C7 implements one and C8 trains it"
      - "therefore 'C6 shortcut gates pass or STOP' is not executable as written"
    why_not_invent_a_bank_probe: >-
      an image-level probe at C6 would create a new feature extractor,
      classifier, split, training budget and seed policy that v1.5 never froze.
    decided_before: >-
      any target access and any valid C6 profile or matched-bank result. No BA
      value was observed to choose this.

    c6_sequence_now:
      - VERIFY_C5_POOL
      - BUILD_SOURCE_REFERENCE
      - FIT_NOMINAL_CALIBRATION
      - BUILD_COMMON_PROFILES
      - EVALUATE_GENERATED_CANDIDATES
      - CHECK_PROFILE_MATCHED_FEASIBILITY
      - SELECT_STRICTEST_PROFILE      # profile irrevocably frozen here
      - BUILD_MATCHED_BANKS
      - VERIFY_C6_LOCKS               # records BA_sep as deferred, never passed
    c6_closure_contract: >-
      profile selected by the STRICT -> NOMINAL -> PERMISSIVE matched-feasibility
      rule alone; profile frozen; three banks of exactly 1024 = 512 Physics + 512
      GPAT; selector V1 invariants; provenance closure; target_access 0. BA_sep is
      neither a selection nor a closure input. The banks ARE scientifically usable
      for C7/C8 source training; only the P3 path stays locked.
    profile_immutability: >-
      a later reliability failure blocks forward progress and never mutates C6
      retrospectively: no reopened profile search, no changed thresholds, no
      changed candidate selection, no C5 re-render, no changed source-domain
      quotas, no rebuilt recipe banks.

    barrier:
      lock: reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
      module: prism_fas.evaluation.detector_reliability
      deadline: after C8 source-only detector training, before C9 closes
        SOURCE_MATRIX_LOCK_C
      c9_precondition: >-
        structural, not a PROJECT_STATE line: C9Adapter.semantic_preconditions
        calls verify_lock, which requires every required test PASSED, the probe
        protocol identity and detector checkpoint identities bound, and
        target_access recorded as 0. UNRESOLVED never counts as a pass, so C10
        and C11 stay unreachable.
      required_tests: [synthetic_vs_real_spoof_probe, residual_scale_zero,
                       recipe_region_shift, artifact_map_swap,
                       cross_route_synthetic, benign_jpeg_corruption,
                       benign_resize_corruption, benign_color_corruption,
                       crop_padding_interpolation]
      canonically_blocked: [benign_glasses_makeup_lowlight]   # never converted to PASSED
      on_failure: >-
        DETECTOR_RELIABILITY_LOCK_C = FAILED blocks C9, C10, C11 and target
        prediction. Negative evidence preserved. No reopened C6, no other C6
        profile, no regenerated C5, no tuned banks, no cherry-picked checkpoints,
        no new probe seeds, no loosened 0.75. Redesign needs a new approved
        protocol version.
      c_h4_distinction: >-
        the hard gate is BA_sep <= 0.75 per arm. C-H4 SUPPORT additionally
        requires BA_sep_LLM below both controls with paired bootstrap CIs, plus
        the validity and recipe-diversity conditions. Passing the gate implies
        nothing about C-H4.

  # --- BLOCKING before the first probe run ------------------------------------
  open_decisions_detector_probe:
    - id: DETECTOR_BA_SEP_PROBE_PROTOCOL
      status: NEEDS_SCIENTIFIC_DECISION
      note: >-
        moving the probe solved the STAGING. The executable protocol is still
        unfrozen: 20 result-affecting fields must be bound before any BA number
        is produced, and none may be chosen after observing a BA value. See
        detector_reliability.PROBE_PROTOCOL_REQUIRED_FIELDS.
    - id: DETECTOR_BA_SEP_EVIDENCE_VECTOR
      status: NEEDS_SCIENTIFIC_DECISION
      finding: >-
        Version-B's evidence vector is p_global, s_region and nine normalized
        regional distances — REGIONAL quantities from a Track-R detector. But
        Version-C Track-R primary rows are DET and LLM only (C-H3), and there is
        no preregistered Track-R RND row, while BA_sep is required for all three
        arms because C-H4 needs BA_sep_LLM < BA_sep_RND.
      forbidden_silent_shortcuts: [add a Track-R RND experiment,
                                   substitute a Track-G vector for RND,
                                   use different feature spaces across arms,
                                   drop RND from BA_sep]
    - id: DETECTOR_BA_SEP_PROBE_SEEDS
      status: NEEDS_SCIENTIFIC_DECISION
      finding: >-
        §3.1.1 says "the three frozen source-only probe seeds" and never names
        them. §18.3 fixes the family 20260806-20260810 for 5-seed rows and the
        first three for 3-seed rows, but that policy is scoped to hypothesis
        TRAINING rows and the probe is not a training row. Inheritance is not
        normative on this audit, and seeds may never be chosen after seeing a BA
        value.

  # --- BLOCKING: the executable probe protocol (superseded, see above) --------
  open_decisions_c6_probe:
    - id: C6_BA_SEP_PROBE_PROTOCOL
      status: NEEDS_SCIENTIFIC_DECISION
      audited_on: 2026-08-23
      finding: >-
        the executable bank-level probe protocol is NOT uniquely recoverable, and
        the one canonical description is incompatible with running at C6 at all.
      evidence:
        - "the only recorded protocol is Version-B reports/m10/
          RELIABILITY_EXECUTION.json, whose acceptance reads: 'held-out balanced
          accuracy of a linear probe on the DETECTOR'S OWN EVIDENCE VECTOR
          (p_global, s_region, nine normalized regional distances)'. p_global and
          s_region are detector tensors (detector/contracts.py), so that probe is
          detector-level and cannot run at C6 — no detector exists until C7."
        - "neither tree contains an executable synthetic-vs-real probe:
          evaluation/reliability.py only DECLARES the test, and the Version-B
          numbers (BA 0.9375, FAILED) came from a driver not in the repository."
        - "§3.1.1 defines BA_sep_arm over THREE frozen source-only probe seeds;
          Version B recorded a single balanced accuracy with no seed, no split
          identity and no training budget, so the three seeds exist nowhere."
        - "running the gate at C6 would need a BANK-LEVEL feature definition
          computed from the images themselves. None is specified or implemented,
          and choosing one would mean inventing a classifier, feature extractor,
          split, training budget and seeds."
      consequence: >-
        the closure gate BLOCKS with
        C6_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION. The Option B sequence
        is implemented and C6 will run through selection, freeze and bank
        construction, then stop at the gate.
      tension_with_the_frozen_decision: >-
        the decision names synthetic_vs_real_spoof_probe as a BANK-LEVEL C6
        closure gate. The only canonical implementation of that test is
        detector-level. Either a bank-level probe protocol must be specified, or
        the gate must move to the detector-level stage.
      decision_needed_from_user: >-
        either (a) specify the bank-level probe protocol — feature definition,
        classifier, split, training budget and the three seed values — or (b)
        move the synthetic-vs-real gate to the detector-level stage alongside
        residual sensitivity, leaving C6 to close on cardinality and matched
        feasibility.

  superseded_open_decisions_c6_reliability:
    - id: C6_RELIABILITY_SEQUENCE
      status: RESOLVED_BY_OPTION_B
      audited_on: 2026-08-23
      question: >-
        which reliability gates are mandatory AT C6 PROFILE-SELECTION time, and
        is the synthetic-vs-real probe (BA_sep <= 0.75) a selection-time gate, a
        post-selection C6 closure gate, or only a pre-target gate?
      spec_evidence:
        - "§11.4: the selected profile must pass (i) the cardinality test and
          (ii) all mandatory source-only shortcut/reliability gates. The
          mandatory set is never enumerated."
        - "§17 table is titled 'Reliability and shortcut gates BEFORE P3 TARGET
          EVALUATION'; the probe's policy reads 'Balanced accuracy SHOULD <=
          0.75; if higher, C6 fails or requires redesign before target'."
        - "C6 stage row: 'shortcut gates pass or STOP', and separately
          'Synthetic-vs-real probe and residual sensitivity run before detector
          target evaluation' — a later deadline than C6 itself."
        - "§3.1.1: C-H4, where BA_sep_arm is defined, 'is evaluated AFTER the
          three final C3 recipe banks and the common C6 synthetic gate are
          frozen'. Read strictly, BA_sep is not available during selection."
      why_it_cannot_be_inferred: >-
        residual sensitivity measures detector decision-score movement and cannot
        run before C7 under any reading, so the mandatory-at-selection set is
        necessarily a subset the spec does not name. And §11.4(ii) and §3.1.1
        point opposite ways for BA_sep specifically.
      result_affecting: >-
        yes. If BA_sep gates selection it can reject STRICT and move the chosen
        profile, which changes the accepted pool and therefore every matched
        bank.
      answers_to_the_six_questions:
        q1_ba_sep_placement: NOT DETERMINED — see the three readings below
        q2_per_profile_ba_sep: >-
          computable in principle, but only by building and probing a matched
          bank per provisional profile, since BA_sep is defined over the matched
          source split. Whether that violates §3.1.1's "after the common C6 gate
          is frozen" is exactly the ambiguity.
        q3_residual_sensitivity: >-
          needs a trained detector, so C7 at the earliest; it must pass before P3
          target evaluation. Not placeable inside C6.
        q4_bank_vs_detector_level: >-
          bank-level, no detector: synthetic_vs_real_spoof_probe.
          detector-level: residual_scale_zero, recipe_region_shift,
          artifact_map_swap, cross_route_synthetic, benign_jpeg_corruption,
          benign_resize_corruption, benign_color_corruption,
          crop_padding_interpolation. benign_glasses_makeup_lowlight is BLOCKED
          for want of a legitimate population.
        q5_anything_executed_today: >-
          no. prism_fas.evaluation.reliability DECLARES ten tests, all PLANNED or
          BLOCKED. C6 executes none and now says so instead of implying a pass.
        q6_artifact_that_prevents_silent_true: >-
          C6_RELIABILITY.json records empty_is_not_a_pass, executed_count,
          required_set_frozen and is_scientific_lock=false; the substage BLOCKS;
          and assess_profile requires an explicit verdict.
      alternatives:
        A_selection_time_gate: >-
          build a provisional matched bank per profile, run the 3-seed probe on
          each, and require BA_sep <= 0.75 for a profile to qualify. Most
          faithful to §11.4(ii); costs three bank builds and three probe runs;
          arguably in tension with §3.1.1.
        B_closure_gate: >-
          select by cardinality + matched feasibility, freeze the profile, build
          the banks, then run the probe once. If BA_sep > 0.75, C6 FAILS. Most
          faithful to §3.1.1 and to §17's "C6 fails" wording; the profile is
          chosen without reliability input.
        C_pre_target_gate_only: >-
          C6 closes on cardinality alone and the probe runs later, before P3,
          alongside residual sensitivity. Matches the §17 table title and the C6
          row's "before detector target evaluation"; weakest reading of "shortcut
          gates pass or STOP" at C6.
      recommendation_if_asked: >-
        B. It satisfies §17's "C6 fails" consequence and §3.1.1's ordering
        without requiring three provisional banks, and it keeps the probe a
        property of the bank that was actually frozen. Recorded as a preference
        only; not implemented.

  # --- reporting-only: the C3 blocker line -----------------------------------
  c3_blocker_line:
    status: STALE_GLOBAL_REPORTING_STATE
    where: src/prism_fas/pipeline/orchestrator.py, the `profile.name == "full"` branch
    finding: >-
      the "C3 live scientific generation remains gated" blocker is appended
      unconditionally for every full-profile invocation and never consults the
      requested stage range, so a scoped `--from C5 --to C6` run prints it too.
      It is static reporting text, not computed state, and it was NOT the cause
      of the C6 calibration failure — that run reached C6
      BUILD_SOURCE_REFERENCE and stopped at the backend construction.
    action_taken: recorded only. No C3 evidence was mutated and the line was not
      suppressed.

  # --- FROZEN: C6_MATCHED_BANK_SELECTOR_V1 -----------------------------------
  c6_matched_bank_selector_v1:
    status: FROZEN
    authorized_by: user, in session, closing C6_MATCHED_BANK_SELECTOR
    frozen_before: any scientific C6 gate or profile outcome was observed
    target_access_at_freeze: 0
    not_derived_from: >-
      the observed 62 C5 semantic failures, arm quality outcomes, q values,
      quality margins, detector results or target data. The same selector would
      have been used for any completed C5 pool.
    dimension_priority:
      0: HARD route cardinality (512 Physics + 512 GPAT per arm)
      1: COMMON source-domain exposure (one quota vector for all three arms)
      2: recipe coverage / exposure balance (soft)
      3: base-live exposure balance (soft)
      4: canonical tie hash, then candidate_id ascending
    source_domain_field:
      manifest_column: dataset        # finalized M3B source_train manifest
      plan_field: live_dataset        # surfaced per frozen C5 plan row
      universe: [casia_fasd, msu_mfsd]
      ordering: canonical ascending domain id
      never_inferred_from: [file paths, filenames, directory names]
    stage_1_planned_target: >-
      per route, ideal[d] = 512 * planned[d] / 1024 over the frozen PRE-GATE C5
      schedule; integer quotas by largest remainder, ties by canonical domain id
      ascending, summing to exactly 512. Never derived from accepted candidates,
      because gate acceptance is a treatment outcome.
    stage_2_common_capacity: >-
      capacity[route,d] = min over RND/DET/LLM of accepted candidates in that
      cell. Quota clipped to capacity; the deficit is refilled one slot at a time
      to the domain maximizing ideal[d] - quota[d] among domains with
      quota[d] < capacity[d], ties by canonical domain id ascending. If
      sum(capacity) < 512 the profile is not matched-bank-feasible for that route.
    stage_3_fill: >-
      per arm, identical algorithm: greedy minimum of
      (recipe_selected_count, live_selected_count, canonical_tie_hash,
      candidate_id) among accepted, unselected candidates in a domain with
      remaining quota.
    recipe_coverage: >-
      2 per recipe per route is a SOFT target reached whenever capacity permits.
      A recipe with fewer accepted candidates is not a failure, is never
      replaced, and its missing exposure is absorbed by the least exposed
      recipes. No manual deficit redistribution, no recipe quality ranking.
    canonical_tie_hash: >-
      SHA256("PRISM_C6_MATCHED_BANK_SELECTOR_V1"|route|source_domain|
      base_position|live_target_sample_id), UTF-8, "|" separator. Excludes arm,
      recipe-generator type, q, every quality score and margin, target
      information, runtime paths and timestamps. Arm-independent by construction
      because all three arms share the frozen C5 schedule.
    quality_ranking: >-
      FORBIDDEN. Once a candidate passes the selected common profile the gate is
      BINARY for subset construction. q remains the §11.2 synthetic
      sample-quality TRAINING WEIGHT and is serialized in the bank for that use
      only; it never affects selection order.
    profile_feasibility_change: >-
      ArmFeasibility (>=512 accepted per route per arm) remains NECESSARY but is
      no longer SUFFICIENT. A profile qualifies only when one identical
      source-domain quota vector can also be constructed for all three arms on
      both routes, and all mandatory reliability gates pass. Engineering
      rehearsal semantics are unchanged.
    on_failure: >-
      STRICT then NOMINAL then PERMISSIVE, retaining evidence at each refusal.
      If none qualifies, C6 scientific FAILS: no widened profile, no arm-specific
      threshold, no altered target distribution, no altered selector, no new
      candidates.
    provenance: >-
      selected + accepted-but-not-selected + rejected + C5 semantic failures
      close the complete set. No loser cleanup.

  # --- FROZEN: C5_RUNTIME_RECOVERY_V1 ----------------------------------------
  c5_runtime_recovery_v1:
    status: FROZEN
    authorized_by: user, in session, closing C5_TRANSIENT_VS_SEMANTIC_GENERATION_FAILURE
    principle: >-
      what a candidate IS is separated from what happened while trying to make
      it. Only a failure proven to be a pure function of the frozen inputs may
      consume a candidate, because a terminal failure is permanent and, under the
      C5 completion contract, one lost candidate fails its whole 2048-arm.
    classes:
      - name: SemanticGenerationFailure
        meaning: deterministic candidate-semantic; would recur identically
        authorized_members: [artifact finalizes to an empty exact mask after uint8
                             quantization]
        behaviour: terminal FAILED_GENERATION, CANDIDATE.json written, retained
                   permanently, never retried, never replaced, arm short by one
      - name: KeyboardInterrupt / SystemExit
        meaning: process interruption, not an outcome
        behaviour: propagate unchanged; no terminal record, not even a runtime
                   attempt; completed candidates preserved; rerun resumes
      - name: RuntimeAttemptFailure
        meaning: everything else - CUDA, OOM, filesystem, codec, unexpected
                 SyntheticBankError
        behaviour: append operational attempt provenance, NO CANDIDATE.json,
                   abort the pass immediately, no later candidate, no in-process
                   retry, no resampling; the next train.py run is the L1 retry
    attempt_record:
      layout: <candidate_dir>/runtime_attempts/RUNTIME_ATTEMPT_<ordinal>.json
      schema: prism-c5-runtime-attempt-v1
      outcome_value: runtime_incomplete
      binds: [candidate_id, generation_identity_sha256, arm, position, route,
              attempt_ordinal, error_type, sanitized_reason, recorded_at_utc,
              diagnostics]
      never_affects: [GenerationIdentity, candidate_id, payload bytes]
    repeated_failure: >-
      attempts accumulate, one file each, and stay non-terminal. Repetition is an
      L0 implementation/environment diagnostic and is never auto-converted into a
      semantic failure.
    orphan_payloads: >-
      CANDIDATE.json is the commit marker and is written last. Payload files
      without it are not completion evidence; the rerun rebuilds the same
      candidate identity and overwrites them.
    adapter_boundary: >-
      C5 RENDER_CANDIDATES catches RuntimeAttemptFailure, writes
      C5_RENDER_INCOMPLETE.json, and returns without reaching
      VERIFY_RAW_CANDIDATES or FINALIZE_C5. No C5_SYNTHESIS_LOCK is written, so
      C6 stays blocked by the strict verifier.

  c5_executor_verified_how: >-
    36 record/plan tests, 36 executor tests, 130 C5 tests in total, all offline.
    The render loop is exercised end to end over fake routes and the REAL
    finalize_discrete, covering resume, corruption rebuild, failure retention and
    the empty-exact-mask refusal.

  scientific_safety:
    version_b_candidate_plan_modified: false
    version_b_synthetic_bank_modified: false
    calibration_bound_into_c5_identity: false
    candidate_bank_generated: false
    target_access: 0
    c4_to_c13_scientific_execution: NOT_RUN

  tests:
    plan_suite: tests/pipeline/test_c5_source_pair_plan.py    # 41 passed
    contract_suite: tests/pipeline/test_c5_candidate_contract.py    # 17 passed
    covers: >-
      cardinality (2048 / 1024+1024 / 4+4 / 2+2), arm independence of the live
      and spoof schedules, exposure fairness within one, source-only reads, the
      three pairing constraints, empty-pool fail-closed, the absence of any
      calibration parameter, every generation-relevant input changing the
      identity, and the Version-B planner remaining untouched.
    pipeline_suite: {passed: 843, failed: 0, skipped: 0}
    broad_regression: {passed: 2299, failed: 7, skipped: 101, seconds: 627.88}
    inherited_failure_set_identical: true

  changed_runtime_files:
    - src/prism_fas/synthesis/c5_source_pair_plan.py    # new

c5_candidate_contract_reconciliation:
  status: CONTRACT_RESOLVED_IMPLEMENTATION_BLOCKED
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session — explicitly a decision gate before implementation
  base_commit: f24baddbcaea9dacd1898d12d0783262334a625d

  # --- RESOLVED: which cardinality contract governs Version-C C5 ------------
  reconciliation:
    verdict: >-
      NOT a contradiction. The two constant sets belong to two different
      experiments. gate_profiles.py is the Version-C contract; candidate_plan.py
      and synthetic_bank_m8.yaml are the Version-B inherited M8 single-bank
      synthesis.
    authoritative_source: >-
      the frozen v1.5 specification, read from the shipped .docx
    version_c_evidence:
      - "§10.4 (the C5 clause): Scientific synthesis budget is fixed at 2048
         candidate renders per arm = 256 recipes x 8 renders/recipe, with exactly
         4 Physics and 4 GPAT candidates per recipe before the common quality
         gate. Final accepted bank is exactly 1024/arm = 512 Physics + 512 GPAT."
      - "§11.3: matched final training banks, exact same cardinality = 1024
         accepted samples/arm; each arm starts from 2048 candidate renders."
      - "§11.4: C6 selects the strictest profile that yields >=1024 accepted in
         EVERY arm from the frozen 2048 candidates/arm."
      - "C5 stage row: Physics/GPAT render for 3 arms; C6 stage row: Gate
         candidates and build exact matched training banks."
      - "executive summary: 384 raw recipe candidates/arm -> 256 recipes/arm;
         2048 synthetic candidates/arm -> 1024 accepted/arm."
      - "shipped C3 banks confirm it independently: det/llm/rnd each hold
         EXACTLY 256 recipes from 384 raw slots. 256 x 8 = 2048."
    version_b_evidence:
      - "configs/synthesis/synthetic_bank_m8.yaml: live_samples 280,
         candidate_recipes_per_live {physics 2, gpat 2}, expected total 1120,
         bank_id_prefix prism_synthetic_bank_m8_v1."
      - "candidate_plan.py hard-codes EXPECTED_PER_ROUTE 560 / EXPECTED_TOTAL 1120."
      - "neither carries an arm dimension anywhere — no 'arm' key, no RND/DET/LLM."
      - "it is keyed on LIVE SAMPLES; the Version-C contract is keyed on RECIPES."
      - "Version-B froze prism_synthetic_bank_m8_v3_e84c78cd2a9b under it."
    decisive_asymmetry: >-
      the strings 1120, 280 and 560 appear NOWHERE in the Version-C v1.5
      specification. Verified by test against the .docx.
    authoritative_for_version_c_c5: gate_profiles.py — 3 arms x 2048 -> 1024/arm
    candidate_plan_py_verdict: >-
      LEGACY / Version-B only. Not directly reusable: it is per-live rather than
      per-recipe, has no arm dimension, and hard-fails on any count but 1120.
      Adapting it would change a frozen Version-B contract, so it is left alone.
    locked_by: tests/pipeline/test_c5_candidate_contract.py    # 17 passed

  # --- NOT RESOLVED: what blocks the C5 executor ---------------------------
  needs_scientific_decision:
    - id: C5_CANDIDATE_IDENTITY_VS_QUALITY_CALIBRATION
      what: >-
        the spec puts rendering at C5 and gating at C6, but the canonical
        SyntheticBankGenerator loads a FrozenCalibration unconditionally and
        binds threshold_sha256, fingerprint_reference_sha256 and
        calibration_sha256 INTO its generation identity. No generation-only mode
        exists.
      why_not_mechanical: >-
        using it at C5 needs a frozen calibration before C6 selects one, which is
        the circular dependency the task forbids and applies a gate the spec
        assigns to C6. Removing those three fields CHANGES candidate identities
        rather than preserving them, and the spec never says whether a Version-C
        candidate identity should bind a quality calibration at all.
      decision_required: >-
        does the Version-C C5 candidate identity bind a quality calibration?
        If not, the canonical generation identity must be re-specified.
    - id: C5_RENDER_TO_LIVE_SAMPLE_MAPPING
      what: >-
        §10.4 says only "use a fixed ordered list of live source sample IDs
        shared across all generator arms" and fixes the budget per RECIPE
        (256 x 8). It does not state how the 8 renders of a recipe are assigned
        to live samples, how long the shared list is, or how it is built.
      why_it_matters: >-
        the candidate identity binds live_target_sample_id, so this mapping
        determines every candidate's identity and its bytes.
      why_version_b_cannot_answer_it: >-
        the M8 construction is per-live (280 x 2 per route), a different shape
        from per-recipe (256 x 8). It is not a lower-level view of the same plan.
      decision_required: >-
        the exact construction of the shared ordered live list and the
        recipe-render -> live-sample assignment rule.

  implementation_status: >-
    NONE. No scientific generation code was written or committed, per the task's
    own instruction that none should be until the gate resolves. The C5
    engineering workflow, the assert_fixture_permitted guards and the C5-C13
    leakage audit from the previous milestones are untouched.
  c4_input_contract_unimplemented: >-
    the C4-lock verification path for C5 was specified in the request but not
    built, because it would be part of the blocked executor.

  scientific_safety:
    scientific_generation_code_committed: 0
    scientific_contracts_changed: 0
    candidate_bank_generated: false
    target_access: 0
    c4_to_c13_scientific_execution: NOT_RUN

  tests:
    contract_suite: tests/pipeline/test_c5_candidate_contract.py    # 17 passed
    reads_the_frozen_spec_directly: true
    pipeline_suite: {passed: 802, failed: 0, skipped: 0}
    broad_regression: {passed: 2258, failed: 7, skipped: 101, seconds: 563.63}
    inherited_failure_set_identical: true

  changed_runtime_files: []      # test and documentation only

c4_scientific_search_closure_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session
  base_commit: c9b7ec11d4c4283a09a7218e9fe454b9b5b0fdc9
  found_by: code audit of c9b7ec1, before the 12 scientific trials were started
  preserved_from_the_previous_milestone: >-
    the scientific/engineering split, the GPATTrainer wiring, the approved LR
    common multiplier, the separated search states and the scientific-only lock
    are all unchanged. This milestone is execution correctness only.

  defect_1_resumed_trials_could_not_be_finalized:
    cause: >-
      `trained` was populated inside evaluate(), and coordinate_search(resume=
      True) reuses a recorded non-INTERRUPTED TrialResult by config SHA WITHOUT
      calling evaluate. The dictionary was therefore empty for exactly the trials
      a resumed run depends on, and _scientific_finalize refused a winner an
      earlier process had legitimately completed.
    fix: >-
      each completed trial writes runs/full/c4/scientific/trial_<sha16>/
      TRIAL_SUMMARY.json, last, after fit() returns, binding the trial config
      SHA, the search-plan identity, the resolved GPAT config hash, the package /
      bank / pair-plan / AdaFace / architecture identities, the best checkpoint
      path and SHA256, the best metrics, epochs and stop reason, the source-only
      audit and the resume lineage. TrialResult.artifacts points at it.
    contract_now: >-
      "valid scientific trial evidence exists and matches this frozen plan",
      NOT "was trained in this pass". Missing or mismatched evidence fails
      closed; nothing accepts metrics from the search state without the run that
      produced them, and no state is deleted to force recomputation.

  defect_2_config_and_checkpoint_could_cross_bind:
    cause: >-
      the finalizer took the checkpoint belonging to payload["winner_config_
      sha256"] and wrote payload["best_config"] beside it. Those are different
      objects.
    authoritative_semantics_audited: >-
      best_config is the coordinate-wise accumulator the pass produces — start at
      the inherited anchor, move one coordinate at a time, carry the winner
      forward — and it is what §15.2.2 defines a coordinate search to yield.
      SearchOutcome.winner is leaderboard()[0], the top row of the ranking of
      INDIVIDUAL trials; a trial from an early coordinate can rank globally best
      while its config lacks every later coordinate's improvement. The project
      already recorded this: the engineering VERIFY_LOCK note says the two
      "coincide only when the last coordinate produced the winner".
    decision: best_config is the scientific C4 selection; the leaderboard winner is diagnostic
    fix: >-
      selected_config = best_config; selected_config_sha256 =
      canonical_config_sha256(selected_config); the trial that evaluated exactly
      that SHA must exist and be finite-valid; its persistent evidence and its
      checkpoint are the ones bound. The leaderboard winner is retained under
      leaderboard_winner_config_sha256 and can never become the selection.
    no_fallback: >-
      if no finite-valid trial corresponds to the final selected config, the
      stage fails closed. It never reaches for the global leaderboard winner.
    verify_lock_fixed: >-
      it compared bool(recomputed), which is true of any input. It now requires
      canonical_config_sha256(selected_config) == selected_config_sha256, and
      that the trial evidence the lock names carries the same config SHA, the
      same checkpoint SHA and the same resolved GPAT config hash.

  defect_3_interrupted_search_could_finalize:
    cause: >-
      _scientific_search accepted outcome.status in ("COMPLETED","INTERRUPTED")
      and returned an outcome either way; the workflow finalized any non-None
      outcome.
    fix: >-
      a non-COMPLETED status returns None with c4_search_completed_before_
      finalization failing and reason_code SEARCH_INCOMPLETE. The state file and
      every trainer checkpoint are preserved, nothing is deleted, and the next
      run resumes at the interrupted trial. No GPAT_CONFIG_LOCK is written.

  strict_cuda_gate:
    was: _scientific_device() returned resolve_device(None), which answers "cpu"
    now: >-
      ScientificDeviceUnavailable is raised unless the device is CUDA. Twelve
      GPAT trials on a CPU would not finish, and if they did they would be
      scientific evidence produced outside the frozen precision.cuda: fp16
      contract. The zero-argument runner already refuses a non-CUDA host at the
      GPU preflight; this is the second lock for the expert path.
    rehearsal_unchanged: true    # only the scientific trial calls it

  scientific_safety:
    scientific_contract_changes: 0
    c4_to_c13_training_executed_on_this_laptop: false
    target_access: 0
    c5_c13_guards_preserved: true

  tests:
    search_closure_suite: tests/pipeline/test_c4_search_closure.py    # 24 passed
    guards_verified_against_each_defect: >-
      each defect was reintroduced in turn and the suites failed — resume 2
      failed, cross-bind 1 failed, interrupt 2 failed — then restored.
    cross_binding_regression: >-
      a leaderboard winner deliberately DIFFERENT from the coordinate-wise best,
      with evidence written for both, asserting the selected config binds its own
      checkpoint.
    routing_suite: tests/pipeline/test_c4_scientific_routing.py    # 30 passed
    leakage_audit: tests/pipeline/test_scientific_fixture_leakage.py    # 25 passed
    pipeline_suite: {passed: 785, failed: 0, skipped: 0}
    broad_regression: {passed: 2241, failed: 7, skipped: 101, seconds: 543.24}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0

  changed_runtime_files:
    - src/prism_fas/pipeline/adapters/c4.py

c4_scientific_execution_routing_closure:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false          # no C4-C13 training ran on this laptop
  authorized_by: user, in session
  base_commit_on_gpu_host: 34c321ee41e654adfe1439afe2f9400c5a9a6734

  # --- the defect ---------------------------------------------------------
  symptom: >-
    the RTX 5090 completed preparation and reported C4 PASS with every mode
    passing — PREPARE_SUPPORT, VALIDATE_SUPPORT, SMOKE_GPAT, SOURCE_SEARCH,
    FINALIZE_GPAT, VERIFY_LOCK — while the stage status stayed sci=NOT_RUN and
    C5 blocked on reports/full/c4/GPAT_CONFIG_LOCK.json being absent.
  root_cause_one_workflow: >-
    C4 had a SINGLE workflow, written as an engineering rehearsal, and
    --profile full executed it. It built a fixture batch, scored trials with
    _identity_stand_in instead of the frozen AdaFace, evaluated each candidate
    with ONE optimizer step, called coordinate_search(require_valid_winner=
    False), wrote C4_ENGINEERING_CONFIG_RECORD.json, and asserted that the
    scientific GPAT_CONFIG_LOCK did NOT exist. Every check passed because the
    engineering path is correct engineering. It was simply in the wrong place.
  root_cause_status_axis: >-
    the scientific axis was hard-coded "NOT_RUN" in TWO places —
    EngineeringAdapter.result and orchestrator._execute_stage — so even a
    genuinely scientific run had no value it could report. That is the other
    half of PASS / sci=NOT_RUN.
  what_was_NOT_done: >-
    the engineering record was not renamed, not copied to GPAT_CONFIG_LOCK, not
    relabelled is_scientific_lock, and the one-step search was not promoted.
    Engineering evidence cannot become scientific evidence by any edit here.

  # --- the fix ------------------------------------------------------------
  routing:
    boundary: ExecutionContext.is_scientific / fixtures_permitted
    rehearsal_path: _engineering_workflow — unchanged, six modes as before
    scientific_path: >-
      _scientific_workflow — _scientific_prepare, _scientific_plan,
      _scientific_search, _scientific_finalize, _scientific_verify_lock. It
      shares no function with the rehearsal, and SMOKE_GPAT is absent by
      construction: a smoke inside a scientific pass would put fixture numbers
      in the same report as scientific ones.
    second_lock: >-
      assert_fixture_permitted() raises FixtureInScientificContext at the
      fixture producers themselves, so re-editing workflow() cannot reopen the
      door.
  inputs: >-
    sources.verify_support_inputs remains authoritative and is unchanged: M3B
    validated, frozen M7 bank, gpat_pairs 896/224, all three identities agreeing.
  trainer: >-
    prism_fas.synthesis.gpat_trainer.GPATTrainer. No training is reimplemented
    in c4.py. The trainer owns SampleStore, bank resolution, both pair
    manifests, real AdaFace, device placement, fp16 AMP, parameter groups,
    scheduler, invariant checks, validation, early stopping, best/last
    checkpoints, checkpoint identity, resume lineage and the source-only audit.
  lr_decision:
    source: configs/search/lr_anchor_decision.yaml via load_decision().for_component("C4")
    identity: 7ef3492263507d4399828089bbe1af79438bc892e50c8ad732585c1d40c8397c
    interpretation: B_common_multiplier
    anchor_vector: {encoder_lr: 2.0e-4, recipe_lr: 1.0e-4, generator_lr: 2.0e-4}
    multipliers: [0.5, 1.0, 2.0]
    ratio_proof_measured_this_session:
      "0.5": {encoder_lr: 1.0e-4, recipe_lr: 5.0e-5, generator_lr: 1.0e-4, ratio: "2.0:1.0:2.0"}
      "1.0": {encoder_lr: 2.0e-4, recipe_lr: 1.0e-4, generator_lr: 2.0e-4, ratio: "2.0:1.0:2.0"}
      "2.0": {encoder_lr: 4.0e-4, recipe_lr: 2.0e-4, generator_lr: 4.0e-4, ratio: "2.0:1.0:2.0"}
    consequence_measured: >-
      with the decision bound the plan has 5 coordinates and 12 trials; without
      it the learning-rate coordinate stays AMBIGUOUS and the plan has 9. The
      two plan identities therefore DIFFER, which is what makes an engineering
      search state unusable as a scientific resume point.
  absent_coordinate_not_invented:
    coordinate: geometry_preservation_weight
    state: ABSENT — gpat_m8.yaml carries no loss.geometry at either declared path
    handling: >-
      §15.2.3 skips an absent scalar and the plan marks it inapplicable. An
      earlier draft of this milestone mapped it onto loss.total_variation, the
      nearest-looking key; that would have invented an inherited anchor and was
      removed. A test asserts the mapping table does not contain it.
  selection_tuple_mapping:
    source: GPATTrainer.validate() — the loss result's detached() already carries
            the invariant errors beside the components, so no metric is invented
    hard_invariant_failure: either invariant over its declared tolerance
    neutral_support_validation_objective: validation_total_loss (= checkpoint_selection.primary)
    identity_drift: validation_identity
    low_frequency_geometry_drift: validation_ll_invariant_max_abs_error
    outside_mask_error: validation_outside_mask_max_abs_error
  search:
    engine: prism_fas.search.coordinate.coordinate_search
    require_valid_winner: true
    on_exhaustion: NEEDS_SCIENTIFIC_DECISION — the envelope is never widened
    one_pass: true
    state_file: reports/full/c4/C4_SCIENTIFIC_SEARCH_STATE.json
    engineering_state_preserved: reports/full/c4/C4_SEARCH_STATE.json — untouched
    per_trial_run_root: runs/full/c4/scientific/trial_<config_sha16>/
    resume: >-
      identity-aware at trial granularity (coordinate_search reuses a recorded
      trial by config hash) and at epoch granularity inside each trial
      (GPATTrainer.fit(resume=True) from its own last.pt). Failed and diverged
      trials are retained in the leaderboard.
  scientific_lock:
    path: reports/full/c4/GPAT_CONFIG_LOCK.json
    schema: c4-gpat-config-lock-v1
    written_only_by: _scientific_finalize, only from a trained winner
    binds: [execution_profile, search_plan_identity, lr_decision_identity,
            lr_interpretation, lr_anchor_vector, selection_tuple, tie_break,
            attempted_config_ids, trials_by_status, winner_config_id,
            winner_config_sha256, selected_config, package_identity,
            recipe_bank_identity, pair_plan_identity, adaface_weight_sha256,
            architecture_hash, config_hash, winning_checkpoint,
            winning_checkpoint_sha256, best_metrics, record_set_hashes,
            resume_lineage, source_isolation, no_target_capability_proof]
    verify_lock_scientific: >-
      re-derives the config identity, hashes the checkpoint on disk against the
      recorded SHA256, and re-resolves the package/bank/pair-plan identities so a
      rebuilt input invalidates the lock instead of silently changing what C5
      inherits. This is the ONE place C4 sets scientific_evidence=True.
    verify_lock_rehearsal: unchanged — verifies C4_ENGINEERING_CONFIG_RECORD.json
  status_semantics:
    adapter: >-
      EngineeringAdapter.result gained scientific_evidence=..., REFUSED with
      StatusError on a profile that is not scientifically eligible, and still
      FAIL when the mode's own checks failed.
    stage: >-
      orchestrator._execute_stage derives the axis from the adapter results
      instead of hard-coding NOT_RUN. NOT_RUN when no mode claimed evidence.

  # --- C5-C13 audit -------------------------------------------------------
  downstream_audit:
    method: >-
      an AST walk over every C4-C13 adapter requiring each fixture producer to
      be guarded by assert_fixture_permitted or to sit inside a branch on
      fixtures_permitted / is_scientific. It runs as a test, so a new unguarded
      fixture callsite fails on this machine.
    guards_added: [c4._prepare_support, c5._render_gpat, c7 complexity fixture]
    stages:
      c5: {scientific_executor: false, full_profile_reaches_it: true,
           fixture_reachable_under_science: false,
           blocker: "_render_gpat builds a fixture batch and a RANDOMLY
                     INITIALIZED generator; its artifact already records
                     trained_checkpoint_used: false, and SMOKE_RECIPES_PER_ARM=2
                     caps the arms. A scientific C5 must render from the C4
                     winning checkpoint over the full candidate budget."}
      c6: {scientific_executor: false, blocker: "SMOKE_CANDIDATES_PER_ARM=8 caps
           the quality-gate calibration."}
      c7: {scientific_executor: false, blocker: "readiness is a CPU fixture
           obligation by design; the scientific detector search is not wired."}
      c8: {scientific_executor: false, blocker: "SMOKE_ROWS=2 caps the rows."}
      c9: {scientific_executor: false, blocker: "reporting over upstream evidence."}
      c10: {scientific_executor: false, blocker: "_fixture_roots builds a
            synthetic target package; the real sealed package is resolved by
            sources._real_target_roots behind the target firewall."}
      c11: {scientific_executor: false, blocker: "prediction rows come from
            adapters.tiny under rehearsal."}
      c12: {scientific_executor: false, blocker: "labels are fabricated by
            adapters.tiny under rehearsal."}
      c13: {scientific_executor: false, blocker: "closure over upstream evidence."}
    conclusion: >-
      C4 is the only stage with a scientific executor. Every other stage can now
      only produce engineering evidence, and none of them can claim scientific
      evidence because none passes scientific_evidence=True — asserted by test.
      Wiring C5-C13 is NOT attempted here; each is recorded with its blocker.

  # --- safety -------------------------------------------------------------
  scientific_safety:
    c4_to_c13_training_executed_on_this_laptop: false
    engineering_artifacts_promoted: 0
    target_access: 0
    models_hyperparameters_search_space_datasets_banks_changed: false
    gpat_config_changed: false
    gemini_calls: 0
    modal_jobs: 0
  honest_limitation: >-
    the scientific C4 executor has NEVER RUN. This laptop is CPU-only, holds no
    preprocessed source package and cannot execute GPAT training, so the wiring
    is asserted structurally — which arguments reach GPATTrainer, which flag
    coordinate_search receives, which artifact each branch writes. Every
    preparation defect in this chain was found by executing rather than reading,
    and the first real execution of this path is on the GPU host.

  tests:
    c4_routing_suite: tests/pipeline/test_c4_scientific_routing.py     # 30 passed
    leakage_audit_suite: tests/pipeline/test_scientific_fixture_leakage.py  # 25 passed
    pipeline_suite: {passed: 761, failed: 0, skipped: 0}
    broad_regression: {passed: 2217, failed: 7, skipped: 101, seconds: 530.30}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0

  changed_runtime_files:
    - src/prism_fas/pipeline/adapters/c4.py
    - src/prism_fas/pipeline/adapters/c5.py
    - src/prism_fas/pipeline/adapters/c7.py
    - src/prism_fas/pipeline/adapters/common.py
    - src/prism_fas/pipeline/orchestrator.py
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_package_lifecycle_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering lifecycle fix
  base_commit_on_gpu_host: 18743c07291e68a26e7c94f644027dc9f87de749
  discovered_by: >-
    the RTX 5090 host reaching C4 PREPARE_SUPPORT. The C4 identity gate added in
    the previous milestone did its job and refused the package.
  symptom: >-
    VERIFY_SUPPORT_INPUTS = FAIL / SourceUnavailable / MISSING_DATA:
    data/packages/prism_data_v1_m3b reports status 'building'; a scientific run
    trains only against a package its own validator passed.

  lifecycle_defect:
    what_step_m3b_did: >-
      returned REUSED_VALID when `target.is_dir()` and PACKAGE_LOCK.json existed,
      and on the build path returned BUILT immediately after build_m3b_package.
    why_locked_is_not_validated: >-
      build_m3b_package WRITES PACKAGE_LOCK.json itself, with status "building".
      The lock exists from the builder's first write; `status` is what says
      whether anything checked it. Presence proves a build started, not that it
      was validated.
    canonical_lifecycle_already_demonstrated_by: >-
      cli/main.py::priors_model_build — build, validate(require_validated_status=
      False, parent_package=M3A), finalize_lock, validate(strict, parent_package=
      M3A). Preparation ran none of it.
    fix: >-
      one helper, preparation.ensure_package_validated, used by BOTH the build
      path and the reuse path of M3A and M3B. It never rebuilds a payload.

  can_an_existing_building_m3b_be_finalized_in_place: yes
  why: >-
    finalize_lock rewrites PACKAGE_LOCK.json and nothing else. The model priors —
    hours of frozen-tower inference — are never recomputed to change a status
    field. A test asserts the payload is byte-identical across finalization.

  identity_consequence:
    finalize_lock_promotes: [status, target_isolation.status, package_validation]
    and_recomputes: content_identity_sha256
    measured_in_test:
      before_finalization: the identity a `building` lock carries
      after_finalization: a different identity, recomputed over the promoted lock
      proof: >-
        test_finalization_changes_the_content_identity asserts before != after
        and that the after value is what package_status reports downstream.
    consequence_for_the_pair_plan: >-
      a plan built against the pre-finalization identity is stale the moment the
      package is finalized. _pair_plan_is_current already compared
      PAIR_PLAN_LOCK.package_identity to the package's content_identity_sha256;
      that behaviour is preserved and now proven, and the plan is rebuilt
      automatically against the finalized identity. No manual deletion.

  second_defect_found_in_my_own_fix:
    what: >-
      what_is_needed / _incomplete decided whether the steps run at all, and
      asked only whether "the marker the builder writes last" was present. The
      builder writes the `building` lock itself, so an unfinalized package looked
      finished and prepare() would have returned NOTHING_TO_DO — the lifecycle
      fix would never have executed on the host.
    fix: >-
      _incomplete now asks the artifact its state: a package must be `validated`
      with package_validation passed, and a pair plan must still be bound to the
      identities that exist now. Caught by auditing the coarse pass rather than
      by another remote round trip.

  nearby_anti_pattern_audit:
    m3a_reuse_branch: >-
      FIXED. It validated with require_validated_status=False, which accepts a
      `building` package. It now uses the same helper.
    other_lock_presence_checks: >-
      the remaining `LOCK.json exists` tests in c4/c5/c7 assert a governing lock
      does NOT yet exist (opposite polarity), and the one in _pair_plan_diagnosis
      is a read-only report field whose `reusable` verdict is identity-based.
      Scope not widened beyond the real instances.

  behaviour_change_recorded:
    what: >-
      a locked package that no longer validates now fails closed with
      PACKAGE_NOT_VALIDATED instead of being silently rebuilt over.
    why: >-
      a package claiming `validated` that stops validating means its content
      changed under a finalized lock. Rebuilding passes over whatever went wrong
      and destroys the evidence of it. test_scenario_e was renamed and rewritten
      to assert the fail-closed contract; this tightens the test, it does not
      weaken it.

  diagnosis_bug_fixed:
    was: >-
      --diagnose-data printed M3B as "locked" whenever PACKAGE_LOCK.json existed.
      That is what hid this defect from the operator.
    now_reports: [present, status, package_validation, content_identity,
                  reusable_as_scientific_input YES/NO, and why not]
    pair_plan_now_reports: >-
      bound_to_current_package, bound_to_frozen_bank, reusable, and — when stale —
      that finalizing M3B recomputes the identity and that no manual deletion is
      needed.
    read_only: true

  scientific_safety:
    package_locks_hand_edited: 0        # finalize_lock is the only promoter used
    payload_files_deleted_or_regenerated: 0
    model_priors_recomputed_to_change_status: 0
    models_hyperparameters_search_space_datasets_banks_changed: false
    target_access: 0
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  tests:
    focused_suite: tests/pipeline/test_package_lifecycle.py    # 23 passed
    focused_covers:
      - a building lock is not scientific input; a validated one is
      - PACKAGE_LOCK presence alone is never reuse
      - an existing building package is finalized in place, payload byte-identical
      - finalization changes the content identity, and that identity is what
        downstream sees
      - the sequence is loose-validate then finalize then strict-validate, and
        both validations receive the M3A parent
      - a validated package is still strict-validated before reuse
      - a building package that does not validate fails closed and is NOT promoted
      - a validated package that stops validating fails closed
      - an unrecognized lock status is refused
      - a newly built M3B is validated, finalized and revalidated
      - the M3A reuse branch no longer accepts a building package
      - a plan built before finalization is not reused; the step rebuilds it
        against the finalized identity and the lock ends bound to it
      - a building package and a stale plan keep their trees on the to-do list,
        so the host state is not reported NOTHING_TO_DO
      - the C4 gate that raised on the host accepts what preparation now finalizes
      - the diagnosis reports status rather than just "locked", explains a stale
        plan, opens no target and writes nothing
      - finalization deletes nothing anywhere
    real_finalize_lock_used: true    # no test manufactures a promoted lock by hand
    orchestration_suite: tests/pipeline/test_preparation.py    # 61 passed
    pipeline_suite: {passed: 706, failed: 0, skipped: 0}
    m3a_m3b_m8_suites: {passed: 158, failed: 2, skipped: 69}   # the 2 are inherited
    broad_regression: {passed: 2162, failed: 7, skipped: 101, seconds: 549.36}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0
    fixture_corrected: >-
      the preparation stub wrote `{"status": "validated"}` as a whole package
      lock. It now writes the real shape — status, content identity,
      package_validation — because a lock without those would let the fixture
      skip the lifecycle these tests exist to exercise.

  changed_runtime_files:
    - src/prism_fas/pipeline/preparation.py
    - train.py
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  manual_cleanup_required: false
  remote_m2_m3a_reuse_expected: true
  remote_m3b_expected: FINALIZED_IN_PLACE_NO_PRIOR_REBUILD
  remote_gpat_pairs_expected: REBUILT_IF_IDENTITY_CHANGED
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_c4_support_input_contract_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering input-contract fix
  base_commit_on_gpu_host: 334b765856a30c258fb20de0ce770870ffc1a62d
  resolves: >-
    the NEEDS_SCIENTIFIC_DECISION recorded by the previous milestone,
    C4_SUPPORT_BATCH_RECIPE_BANK. It is settled by the code contract rather than
    by preference: m8_pipeline.build_batch does `recipes[pair["recipe_id"]]` over
    the bank it is handed, so the support batch MUST resolve the same bank the
    pair plan drew its recipe ids from, or the lookup is a KeyError. The pair
    plan is identity-bound to M3B + M7, and the frozen M8 evidence binds the same
    two. C3 was never a candidate for this path — it is the rehearsal fixture's
    conditioning source, and no C3 root satisfies load_bank at all.

  defect_1_package_root_was_the_parent_directory:
    bad_path_before: data/packages
    canonical_path_after: data/packages/prism_data_v1_m3b
    why: >-
      SampleStore.open reads <package_root>/manifests/source_train.parquet.
      `data/packages/manifests/` has never existed; the parent directory is the
      container the asset inventory declares, not a package.

  defect_2_bank_root_was_the_c3_container:
    bad_path_before: assets/recipe_banks/c3
    canonical_path_after: assets/recipe_banks/prism_recipe_bank_m7_v1
    why: >-
      m8_pipeline.resolve_bank IS recipes.bank.load_bank. The C3 container holds
      the three treatment banks (det/llm/rnd) and carries none of the seven
      BANK_FILES; no arm satisfies it either. Measured in the previous milestone
      and re-measured here.

  single_declaration: >-
    adapters/sources.py now derives SOURCE_PACKAGE_ROOT, GPAT_PAIR_ROOT and
    RECIPE_BANK_ROOT from preparation.DERIVED_PACKAGES / PAIR_PLAN_PACKAGE /
    M7_RECIPE_BANK — the module that PRODUCES those artifacts. Producer and
    consumer cannot drift apart, and a rename propagates.

  identity_gate:
    function: prism_fas.pipeline.adapters.sources.verify_support_inputs
    runs_before: any tensor is built
    raises:
      SourceUnavailable: MISSING_DATA — an input is absent or unvalidated
      SupportIdentityMismatch: IDENTITY_MISMATCH — all present and disagreeing
    checks:
      - the pair-plan lock and BOTH pair manifests exist
      - the M3B PACKAGE_LOCK exists, status == validated
      - package_validation.status == passed
      - package content identity == the plan's package_identity
      - the M7 bank passes preparation.validate_recipe_bank (BANK_FILES,
        validate_bank, status frozen, bank id, pinned content identity)
      - that bank identity == the plan's recipe_bank_identity
      - every recipe id in the pairs actually used exists in that bank, checked
        before build_batch rather than surfacing as a bare KeyError inside it
    opens: three lock files; no manifest, no image, no target
    writes: nothing

  identity_agreement_measured_this_session:
    shipped_m7_bank: {bank_id: prism_recipe_bank_m7_v1, status: frozen, recipes: 128}
    m7_identity: fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb
    m3b_identity: b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6
    frozen_version_b_pair_plan: {train_pairs: 896, validation_pairs: 224, seed: 20260806}
    bank_identity_equals_frozen_plan_bank_identity: true
    preparation_pin_equals_shipped_bank_identity: true

  source_only_contract:
    manifests_opened: ["manifests/source_train.parquet"]
    source_dev_opened: false
    target_test_opened: false
    measured_how: >-
      the SampleStore's own SourceOnlyAudit records every relative path it opens
      and refuses a forbidden split or a target path at record() time. Its report
      is now carried in the C4 support provenance, so isolation is a measurement
      the artifact holds rather than a claim. The fixture additionally writes
      source_dev and target_test as unparseable bytes.
    target_access: 0

  rehearsal_contract_preserved:
    branch_unchanged: true
    conditioning_source: assets/recipe_banks/c3
    fixture_backed: true
    made_explicit: >-
      the rehearsal provenance now carries `conditioning_source` and says in
      words that C3 conditions the REHEARSAL only. No rehearsal behaviour changed.
    scientific_fallback_possible: false   # fixtures_permitted = not scientific_eligible

  scientific_safety:
    banks_or_packages_repaired_rewritten_or_regenerated: 0
    aliases_created: 0
    c3_files_copied_into_m7: 0
    recipes_regenerated: 0
    external_llm_called: false
    assets_bytes_changed: 0        # `git status --porcelain assets/` is empty
    target_access: 0
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  nearby_paths_audited:
    every_samplestore_open_callsite: >-
      the other callers live in synthesis/ and take package_root as a parameter;
      the pipeline supplies it only here. No GPATTrainer is constructed anywhere
      in the pipeline, so there is no second stale trainer root.
    remaining_data_packages_parent_references: >-
      RequiredInput presence declarations in c4/c5/c8, the assets and handoff
      inventories, portable_paths roots and preparation.diagnose — all correct as
      container-level declarations; none is passed to SampleStore.open.

  tests:
    focused_suite: tests/pipeline/test_c4_support_contract.py    # 26 passed
    guards_verified_to_fail_against_the_defect: 22   # reverted both roots, then restored
    focused_covers:
      - the sample store is opened on the M3B package, never on the parent
      - the bank resolved is the frozen M7 bank; no C3 root or arm under science
      - the roots come from the producer, not a second spelling
      - the plan's package identity equals the loaded M3B identity
      - the plan's bank identity equals the loaded M7 identity
      - build_batch receives recipe ids that exist in that exact bank
      - a plan from another bank, another package, an unvalidated package, an
        incomplete plan and an unfrozen bank are each refused before anything opens
      - the gate writes nothing
      - only source_train.parquet is opened; source_dev and target_test are not
      - the audit refuses a forbidden open by construction
      - no target artifact is named anywhere in the provenance
      - the rehearsal still uses the C3 fixture conditioning and stays fixture_backed
      - a scientific ExecutionContext cannot take the fixture branch
    pipeline_suite: {passed: 683, failed: 0, skipped: 0}
    m7_m8_loader_suites: {passed: 227, failed: 2, skipped: 88}   # the 2 are inherited
    broad_regression: {passed: 2139, failed: 7, skipped: 101, seconds: 553.91}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0

  changed_runtime_files:
    - src/prism_fas/pipeline/adapters/sources.py
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  manual_cleanup_required: false
  remote_m2_m3a_m3b_reuse_expected: true
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_gpat_frozen_bank_contract_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering bank-contract fix
  base_commit_on_gpu_host: 643714a1e376665c6f2afff838deb9f577c7791c
  discovered_by: >-
    the real Linux RTX 5090 host, which after the M3B config fix completed M2,
    M3A and M3B and then stopped at the fourth and last preparation step.
  symptom: >-
    [PREPARATION_FAILED] derived-data preparation failed at gpat_pairs:
    BankError: assets/recipe_banks/c3 is not a frozen recipe bank; missing
    ['recipes.jsonl','ontology.yaml','prompt.txt','generator.json',
    'coverage.json','validation.json','BANK_LOCK.json']. Stopped BEFORE C4.

  defect_1_c3_container_passed_to_load_bank:
    bad_path_before: assets/recipe_banks/c3
    canonical_path_after: assets/recipe_banks/prism_recipe_bank_m7_v1
    why_c3_is_not_a_valid_load_bank_input: >-
      it is a CONTAINER of the three C3 scientific banks (det/, llm/, rnd/), each
      holding C3_BANK.json + recipes.jsonl. It carries none of the seven files
      recipes.bank.BANK_FILES requires, and neither does any arm inside it — both
      were measured, not assumed. The C3 banks and the frozen M7 bank are
      different contracts; PORTABLE_ASSET_MANIFEST.json lists the C3 container
      under its own logical name `c3_scientific_recipe_banks`, and nothing in
      this fix converts, copies or regenerates either.
    authority: >-
      frozen Version-B evidence (authority level 1). The Version-B pair-plan lock
      reports/m8/pairs/PAIR_PLAN_LOCK.json records recipe_bank_identity =
      fa989938…10cb, and docs/c0/C0_VERSION_B_INTEGRITY.md §2.2 records the same
      bank. Both production M8 callsites — modal_m8.py and
      cli/main.py::_m8_defaults — already pass this root.

  defect_2_wrong_package_root:
    found_by: >-
      auditing the other argument of the same call rather than stopping at the
      one that raised. It had not fired yet and would not have raised at all.
    bad_path_before: data/packages/prism_data_v1_m3a
    canonical_path_after: data/packages/prism_data_v1_m3b
    why_it_matters: >-
      M3A is structurally loadable by load_source_train_rows, so the plan would
      have been BUILT successfully with the wrong `package_identity` — a value
      stamped into every pair_id and into pair_plan_identity_sha256. A silent
      scientific-identity error, not a crash.
    authority: >-
      the frozen Version-B pair-plan lock records package_identity =
      b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6, which
      docs/c0/C0_VERSION_B_INTEGRITY.md §2.1 identifies as prism_data_v1_m3b.
      modal_m8.py REMOTE_PACKAGE and cli/main.py::_m8_defaults both use M3B.

  bank_validation_gate:
    function: prism_fas.pipeline.preparation.validate_recipe_bank
    runs_before: any pair-plan construction
    reason_code_on_failure: RECIPE_BANK_INVALID
    checks:
      - every file in recipes.bank.BANK_FILES exists
      - the canonical validate_bank re-derives every hash in BANK_LOCK.json
      - BANK_LOCK status == frozen
      - bank_id == prism_recipe_bank_m7_v1
      - bank_content_identity_sha256 == the frozen project contract
    why_the_pinned_identity: >-
      validate_bank alone is satisfied by ANY internally consistent frozen bank,
      so it catches a tampered bank but not a substituted one. Only the pinned
      identity refuses the wrong bank, which is the defect class this gate exists
      for. The pin records a frozen fact; it does not choose one.
    writes: nothing; it never repairs a bank
    m7_bank_identity_before: fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb
    m7_bank_identity_after: fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb
    bank_bytes_changed: 0        # `git status --porcelain assets/` is empty

  resume_semantics:
    rule: >-
      a pair plan is REUSED_VALID only when PAIR_PLAN_LOCK.json exists, BOTH pair
      manifests exist, and the lock's recipe_bank_identity and package_identity
      match the bank and package this run resolved. Presence of the lock alone
      proved the write finished, not that it finished against these inputs.
    interrupted_plan_is_reused: false
    plan_from_another_bank_is_reused: false
    plan_from_another_package_is_reused: false

  source_only_contract:
    manifests_opened: ["manifests/source_train.parquet"]
    source_dev_opened: false
    target_test_opened: false
    measured_how: >-
      pyarrow.parquet.read_table is instrumented in the regression and the
      recorded open list is asserted to be exactly source_train.parquet; the
      fixture writes source_dev and target_test as unreadable bytes so opening
      one would be a hard failure.
    frozen_pair_counts: {train: 896, validation: 224}
    counts_enforced_by: >-
      write_pair_plan raises PairPlanError before writing anything; a regression
      asserts the constants, the gpat_m8.yaml declaration and the frozen
      Version-B lock all agree, and that a short plan leaves no lock behind.

  # --- found, verified, NOT changed -----------------------------------------
  needs_scientific_decision:
    id: C4_SUPPORT_BATCH_RECIPE_BANK
    where: src/prism_fas/pipeline/adapters/sources.py::_real_support_batch
    observed: >-
      bank_root = repo / "assets/recipe_banks/c3", passed to
      m8_pipeline.resolve_bank, which IS recipes.bank.load_bank. Measured: it
      raises the identical BankError, and so does every C3 arm. This path runs
      only when fixtures are not permitted — i.e. exactly the GPU scientific
      path — so it is the next thing that fails on that host, at C4.
    why_not_fixed_here: >-
      unlike the pair-plan roots, no frozen artifact says which recipe bank
      conditions the Version-C C4 GPAT support batch. Choosing between the frozen
      M7 bank and re-serializing a C3 arm into the M7 contract is a scientific
      decision outside a preregistered envelope, so it stops here and is reported
      rather than guessed (CLAUDE.md STOP conditions).
    blocking: C4

  scientific_safety:
    recipe_contents_changed: false
    recipe_identities_changed: false
    c3_banks_modified: false
    m7_bank_modified: false
    bank_regenerated: false
    banks_converted_between_contracts: false
    files_created_inside_assets_recipe_banks_c3: 0
    external_llm_called: false
    search_space_lr_models_datasets_gpat_hyperparameters_changed: false
    target_access: 0
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  tests:
    focused_suite: tests/pipeline/test_gpat_pair_plan_contract.py   # 27 passed
    guards_verified_to_fail_against_the_defect: 13   # reverted both roots, then restored
    focused_covers:
      - the step resolves the frozen M7 bank, not the C3 container
      - load_bank succeeds on the exact shipped root; validate_bank passes
      - every BANK_FILES entry is present
      - the C3 container is not a frozen bank (measured)
      - no C3 arm (det/llm/rnd) can be silently substituted
      - the C3 banks keep their own contract and grew no M7 files
      - the gate reports the frozen identity and writes nothing
      - a missing bank, a substituted-but-consistent bank and an unfrozen bank
        are each refused with RECIPE_BANK_INVALID
      - the plan is bound to the M3B package
      - the frozen pair counts agree across the constants, gpat_m8.yaml and the
        Version-B lock
      - only source_train.parquet is opened; source_dev and target_test are not
      - no target artifact is touched
      - the lock is written only after a successful construction
      - reuse is refused for an interrupted plan, another bank, another package
      - the step hands the builder the two frozen roots, and validates the bank
        before the builder runs
    orchestration_suite: tests/pipeline/test_preparation.py   # 61 passed
    pipeline_suite: {passed: 657, failed: 0, skipped: 0}
    m7_m8_suites: {passed: 225, failed: 2, skipped: 72}   # the 2 are inherited
    broad_regression: {passed: 2113, failed: 7, skipped: 101, seconds: 599.36}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0
    fixture_corrected: >-
      the preparation fixture now copies the REAL frozen M7 bank (137 kB) rather
      than faking one, so the gate does real work in the stub suite too, and the
      chaining test asserts M3B + the M7 bank instead of the values it used to
      encode.

  changed_runtime_files:
    - src/prism_fas/pipeline/preparation.py
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  manual_cleanup_required: false
  remote_m2_m3a_m3b_reuse_expected: true
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_m3b_config_path_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering config-path fix
  base_commit_on_gpu_host: 0ce123cbec319e5cf637ce1fc894f20224c5df65
  discovered_by: >-
    the real Linux RTX 5090 host, which after the M2/M3A contract fix completed
    M2 preprocessing and the whole M3A package and then stopped at the third
    preparation step.
  symptom: >-
    [PREPARATION_FAILED] derived-data preparation failed at m3b_priors:
    FileNotFoundError: configs/models/model_priors.yaml. Stopped BEFORE C4.

  root_cause: >-
    `_step_m3b` passed `repo / "configs" / "models" / "model_priors.yaml"` to
    build_m3b_package, which opens it through load_model_config. That file has
    never existed in this repository under any commit. The canonical M3B
    model-prior config is `configs/models/m3b_priors.yaml`, which is what
    src/prism_fas/cli/m10_target.py, tests/test_m3b_model_priors.py,
    docs/M8_QUALITY_GATE_CONTRACT.md, docs/M10_TARGET_DATA_CONTRACT.md and the
    CHANGELOG all reference. It was a name written once at a call site and never
    compared to the folder.
  why_tests_missed_it: >-
    the preparation fixture wrote a placeholder at
    `configs/models/model_priors.yaml`, so the stub suite created the file the
    production string named. The same shape as the previous two defects: a
    fixture that agreed with the code instead of with the repository.
  fix: >-
    `_step_m3b` names the canonical config. No alias file was created, no
    scientific config content changed, and configs/models/m3b_priors.yaml is
    byte-unchanged.
  generalised: >-
    every config the preparation path hands a builder is now declared once in
    preparation.PREPARATION_CONFIGS (plus DATASET_CONFIG_TEMPLATE for the
    per-dataset adapter definitions) and referenced from there, so a name lives
    in one checkable place rather than at four call sites.

  repository_wide_audit:
    method: >-
      AST walk over src/prism_fas/**/*.py and train.py, reconstructing both
      whole-string paths and `repo / "configs" / ... / "x.yaml"` division chains,
      checked against the 54 files under configs/ and assets/.
    stale_runtime_paths_found: 1        # configs/models/model_priors.yaml, fixed
    remaining_absent_but_correct:
      - path: configs/data/target_layout.yaml
        named_by: src/prism_fas/pipeline/adapters/sources.py
        verdict: NOT_A_DEFECT
        why: >-
          it declares where the sealed held-out target package lives and is
          produced by C10 target preparation, which has never run. The call site
          checks is_file() and raises SourceUnavailable with its own message
          rather than opening it, and it cannot be reached before C4.
    other_config_paths_verified_present: >-
      m9_detector.yaml, m9_reference.yaml, loader_m4.yaml, ontology_m7.yaml,
      c2c_route_policy.yaml, m10_matrix.yaml, gpat_m8.yaml, quality_gate_m8.yaml,
      package_m3a.yaml, preprocess_m2.yaml, m2_run_profiles.yaml,
      casia_fasd.yaml, msu_mfsd.yaml, paths.local.yaml

  scientific_safety:
    scientific_config_contents_changed: false
    alias_or_duplicate_config_created: false
    m2_or_m3a_artifacts_touched: false
    frozen_configs_changed: false
    c3_banks_changed: false
    target_access: 0
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  tests:
    new_guards:
      - >-
        test_every_config_preparation_names_exists_in_the_shipped_folder — walks
        PREPARATION_CONFIGS and the per-dataset configs against the real folder.
      - >-
        test_no_call_site_spells_a_config_path_of_its_own — a path spelled inline
        is a path no test can walk.
      - >-
        test_the_m3b_step_is_given_the_canonical_model_prior_config — asserts the
        canonical name reaches the builder AND that no alias file was added.
      - >-
        test_m3b_reaches_its_builder_without_a_missing_config — copies the real
        config into the fixture, lets _step_m3b open it through the real
        load_model_config, and asserts the keys build_m3b_package reads.
      - >-
        test_every_config_the_runtime_names_actually_exists — the repository-wide
        audit, over the whole runtime tree rather than preparation alone.
      - >-
        test_a_config_that_does_not_exist_yet_is_guarded_not_opened — keeps the
        single target_layout.yaml exemption honest: it must still be absent, still
        be referenced, and its call site must still check is_file().
    guards_verified_to_fail_against_the_defect: true   # reverted the name, 4 failed, restored
    fixture_corrected: >-
      the preparation fixture no longer writes configs/models/model_priors.yaml;
      it names both configs from PREPARATION_CONFIGS.
    preparation_suite: {passed: 61, failed: 0, skipped: 0}
    firewall_suite: {passed: 40, failed: 0, skipped: 0}
    pipeline_plus_m3b: {passed: 642, failed: 0, skipped: 3}
    broad_regression: {passed: 2086, failed: 7, skipped: 101, seconds: 525.07}
    inherited_failure_set_identical: true
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0

  changed_runtime_files:
    - src/prism_fas/pipeline/preparation.py
  handoff_package: none built; one runtime file changed and git is the route
  superseded_handoff_copy:
    path: reports/handoff/LINUX_RTX5090_M2_M3A_CONTRACT_HOTFIX/src/prism_fas/pipeline/preparation.py
    note: >-
      still byte-correct for the commit its manifest names (eade289) and left
      unchanged, but it predates this fix. Anyone applying that package's
      MANUAL FALLBACK would reintroduce the M3B config path. Its README already
      says git is the preferred route; use the NEW_HEAD below.
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  manual_cleanup_required: false
  m2_m3a_rework_required: false     # both completed on the host and are reused
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_m2_m3a_contract_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering derived-data preparation fix
  base_commit_on_gpu_host: 787b9b28367df039a6b3805a7871f92c60a343ae
  discovered_by: >-
    the real Linux RTX 5090 host reaching derived-data preparation for the first
    time in this project's history and running SCRFD over the source corpora.
    Not reachable on the build machine: every laptop rehearsal takes
    `dry_run=not plan.is_scientific`, so the real orchestration branch had never
    executed anywhere.
  symptom: >-
    [PREPARATION_FAILED] derived-data preparation failed at m3a_package:
    FileNotFoundError: <project>/data/processed/manifests/source_frames.parquet.
    Stopped BEFORE C4; no scientific work started.

  defect_1_producer_consumer_path_mismatch:
    producer_before: >-
      preparation._step_m2 -> m2_runner.run(dataset, ...), the legacy CLI helper,
      which writes JSONL results and crops under
      <work_root>/m2/<version>/<config_hash>/m2a/ and never writes a parquet
      manifest at all.
    consumer_before: >-
      preparation._step_m3a -> build_package(paths.processed_root, ...) ->
      load_m2_samples, which reads <input_root>/manifests/{source,target}_
      {frames,crops}.parquet.
    root_cause: >-
      two independent path expressions for one artifact. data/processed has never
      held an M2 manifest under any convention this project has used: in the
      inherited Version-B layout it holds built PACKAGES (prism_data_v1_m3b,
      prism_target_eval_v2, the M8 synthetic banks) and Version-C preparation
      writes packages to data/packages. Nothing has ever written
      data/processed/manifests/.
    why_tests_missed_it: >-
      the preparation suite stubbed m2_runner.run with a fake that created
      data/processed/<dataset>/. The stub wrote where the consumer read, so
      producer and consumer agreed only inside the test. build_package was
      stubbed too, so load_m2_samples never ran. Both sides of a contract were
      replaced by fixtures that agreed with each other and with nothing else.
    fix: >-
      both sides resolve the location through one function,
      preparation.m2_output_root(repo), which delegates to the project's own
      run_profiles.profile_root under the full_preprocessing profile. The
      producer is now the production run_preprocessing + PreprocessingRunContext
      pair — the same pair `prism data preprocess run --run-profile
      full_preprocessing` drives.
    authoritative_m2_path: >-
      <work_root>/m2/<preprocessing_version>/<config_hash>/full_preprocessing
    why_that_basename: >-
      build_lock records source_m2_namespace = input_root.name beside a
      hard-coded source_m2_validation_profile = "full_preprocessing", so any
      other basename produces an M3A lock that contradicts itself.
    also_moved: >-
      build_preprocessing_run_context moved from cli/main.py to
      data/run_context.py. The preparation path could not reach the canonical
      constructor without importing the whole CLI, which is why it drifted onto
      the legacy runner; the CLI now imports it from its new home and its body is
      unchanged.

  defect_2_completeness_was_directory_presence:
    found_by: auditing the reuse rule rather than waiting for it to fire
    root_cause: >-
      _step_m2 returned REUSED_VALID when data/processed existed and was
      non-empty — equally true of a tree that died halfway through its first
      dataset, which M3A would then have consumed.
    canonical_completion_rule:
      - all five canonical manifests present under <m2_root>/manifests/
      - no target rows (source-only before C4; a SiW row is a firewall breach)
      - source_frames row count equals source_crops row count
      - every canonical source record walked, measured per (dataset,
        source_record_id) over source_frames plus preprocessing_failures
      - a completion marker at state/M2_PREPARATION_COMPLETE.json whose
        preprocessing_config_hash, detector_model_sha256 and per-dataset record
        counts still match what the adapters report now
      - the canonical validate_full_profile (m2f1a-full-v1) passes — crop
        SHA-256s, decodability, dimensions, orphans, temporaries, target
        isolation, and the pinned detector hash
    validator_reused_not_reinvented: prism_fas.data.m2_validation.validate_full_profile

  defect_3_m3a_validated_before_finalize:
    found_by: running M2 -> M3A end to end for the first time, in a test
    root_cause: >-
      build_package writes PACKAGE_LOCK.json with status "building"; _step_m3a
      then called validate_package(root) with require_validated_status defaulting
      to True, which can never pass before finalize_lock runs.
    fix: validate loose, finalize, validate strict — the order the canonical CLI uses

  resume_semantics:
    unit: canonical record (video), keyed on source_record_id
    settled_when: >-
      the record contributed at least frames_per_video rows across source_frames
      and preprocessing_failures. A failure row counts: a frame that was walked
      and routed is finished work, and re-walking it every resume would never
      terminate on a corpus with an undecodable file.
    known_residual: >-
      a video so short that uniform_indices yields fewer than frames_per_video
      samples is re-walked on every resume. Deterministic, idempotent and bounded
      to a handful of frames; the alternative — trusting a count we cannot verify
      without opening the media — would let a genuinely truncated record be
      called done.
    nothing_is_deleted: true
    operator_cleanup_required: false

  legacy_m2a_artifacts:
    reusable_as_m2_input: false
    why: >-
      the legacy namespace holds JSONL results, not the canonical parquet
      manifests M3A reads, and migrate_m2a is contract-locked to the frozen
      24/24/12/12/0 acceptance counts (m2b1a-v1), so it cannot carry a full
      corpus across. Adopting it would mean inventing a migration for artifacts
      whose provenance the contract does not cover.
    action_taken: >-
      none. Nothing deletes it and nothing adopts it. `train.py --diagnose-data`
      names it, counts it and says plainly that it is not reusable.
    consequence: >-
      the SCRFD detection the failed remote run performed into the m2a namespace
      is not reused; the full profile redoes it into its own namespace and
      resumes from there onward. Stated rather than hidden.

  operator_forensics:
    command: /usr/bin/python3 train.py --diagnose-data
    report: reports/preflight/DERIVED_DATA_DIAGNOSIS.json
    reports_on: [m2 config root, namespaces present, legacy m2a root and crop
                 count, full_preprocessing root, per-manifest row counts, crops on
                 disk, completion marker, M2 status and outstanding records,
                 package and pair-plan state, what data/processed is and is not]
    read_only: true

  new_reason_codes:
    M2_INCOMPLETE: >-
      the M2 tree did not reach completion. Partial work is preserved and a rerun
      continues from it.
    TARGET_IN_SOURCE_TREE: >-
      a target row appeared in the source M2 tree. Not repairable by preprocessing
      more of it; the run stops for investigation.

  scientific_safety:
    scientific_configs_changed: false
    frozen_configs_changed: false
    datasets_changed: false
    c3_banks_changed: false
    lr_or_search_decisions_changed: false
    scrfd_frozen_policy_changed: false
    partial_artifacts_deleted: 0
    m3a_validation_weakened: false
    target_access: 0
    siw_mv2_preprocessed: false
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  tests:
    focused_suite: tests/pipeline/test_m2_m3a_contract.py    # 24 passed
    focused_method: >-
      nothing stubs the write location. run_preprocessing, ManifestRepository,
      the routing converters, the real media readers, load_m2_samples,
      build_package, validate_package and finalize_lock are all the shipping
      code; only the ONNX session is replaced, and the real frozen SCRFD file is
      still resolved and hashed so validate_full_profile's pinned-digest check
      stays switched on.
    focused_covers:
      - fresh scientific preparation from raw media to a validated M3A package
      - the producer writes the manifests the consumer reads
      - crop_relative_path resolves from the root M3A is given
      - producer and consumer resolve the same root, named full_preprocessing
      - the canonical root is never data/processed
      - non-empty but incomplete is not complete
      - a missing source_frames.parquet / source_crops.parquet is not complete
      - an interrupted run names the records that remain
      - a stale completion marker is not believed
      - a tampered crop fails the canonical validator
      - completed records are reused, not reprocessed
      - partial work survives a failed run
      - a complete validated tree is REUSED_VALID, and a second pass is byte-identical
      - M3A refuses an incomplete or tampered M2 tree
      - no target dataset is preprocessed and no SiW artifact appears in the tree
      - target rows in the source tree stop the run with their own reason code
      - a tree the validator refuses stops the run and is not deleted
      - the diagnosis separates the two namespaces and adopts neither
    orchestration_suite: tests/pipeline/test_preparation.py  # 57 passed
    stop_before_c4_tests: >-
      M2_INCOMPLETE and TARGET_IN_SOURCE_TREE each reach the operator by name
      on stderr, end the run with EXIT_BLOCKED and 'Stopped BEFORE C4', and
      let no stage start.
    pipeline_suite: {passed: 622, failed: 0, skipped: 0}
    broad_regression: {passed: 2080, failed: 7, skipped: 101, seconds: 588.16}
    inherited_failure_set_identical: true   # test-id by test-id vs reports/c0/C0_TEST_SUITE.json
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0
    test_changes_that_were_not_weakening:
      - >-
        test_gpu_hotfix_package now compares the delivered bootstrap package
        against the commit it names rather than the working tree. A later
        milestone editing train.py does not make a delivered package wrong; it
        would only be wrong if its bytes stopped matching the commit it declares.
      - >-
        test_train_py_delegates_rather_than_implementing allows _diagnose_data,
        which is console formatting over preparation.diagnose and owns no
        pipeline behaviour — the same category the test's own docstring already
        permits.

  handoff_package: reports/handoff/LINUX_RTX5090_M2_M3A_CONTRACT_HOTFIX/
  changed_runtime_files:
    - src/prism_fas/pipeline/preparation.py
    - src/prism_fas/data/run_context.py
    - src/prism_fas/cli/main.py
    - train.py
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  manual_cleanup_required: false
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

linux_rtx5090_autograd_preflight_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as an engineering preflight hotfix
  base_commit_on_gpu_host: 1f2e24fa6a8c6eddf465c0e55cde6897cf4b5ba2
  discovered_by: >-
    the real remote Linux RTX 5090 host, running the deployed copy. Not
    reproducible on the build machine by accident: the probe that failed only
    executes when torch.cuda.is_available(), and this laptop carries the CPU wheel.
  host:
    os: Linux
    python: 3.12.3
    torch: 2.13.0+cu129
    gpu: NVIDIA GeForce RTX 5090
    vram_mb: 32607
    driver: 591.86
    compute_capability: "12.0"
  gates_passed_before_the_failure:
    - environment: cuda-cu129
    - device: CUDA
    - bundle: PASS 24/24
    - disk: PASS
    - write_access: PASS
    - target_firewall: ARMED
    - execution_intent: GPU_SCIENTIFIC_FULL
    - stage_range: C4 -> C13

  defect_1_autograd_probe_wrong_input_contract:
    symptom: >-
      [AUTOGRAD_FAILED] the representative model could not complete a
      forward/backward step: AttributeError: 'Tensor' object has no attribute
      'image'. Stopped BEFORE C4; no scientific work started.
    root_cause: >-
      gpu_preflight._model_roundtrip built the real PRISMDetector under
      TRACK_R_FLAGS and then called it with torch.randn(2, 3, 224, 224).
      PRISMDetector.forward takes a DetectorBatch and reads batch.image and
      batch.region_priors on its first two lines. The probe had invented an input
      contract the trainer does not use, so it could only ever have passed by
      accident.
    incorrect_old_input_type: torch.Tensor [2,3,224,224]
    expected_input_type: prism_fas.detector.contracts.DetectorBatch
    why_it_was_never_caught_here: >-
      the probe body is unreachable without CUDA. The CPU rehearsal path returns
      from run_preflight at the cuda_is_available gate with strict=False, so no
      test and no rehearsal on this machine had ever executed those lines.
    fix: >-
      the probe now builds its batch through the SAME public contract the trainer
      uses — batch_contract_for("G5", M9TrainingConfig(...)) for the composition,
      audit_batch(variant, contract) for the tensors, then .to(device) — and runs
      the real loss graph through compute_losses with enabled_terms("G5", variant).
      The scientific model's forward API was NOT changed to accept a bare tensor.
    proves_before_c4:
      - forward completes on the selected CUDA device
      - the loss is a scalar and finite
      - backward completes
      - every trainable parameter receives a gradient (not "some parameter")
      - every gradient is finite
      - batch, output, loss, parameters and gradients are all on the selected device
      - no silent CPU fallback
    writes: nothing
    opens_dataset: false
    resolves_target: false

  defect_2_audit_stub_tower_stayed_on_the_host:
    found_by: >-
      tracing the device semantics of the corrected path rather than waiting for
      the GPU host to fail a second time. It had not fired yet; it was the next
      thing that would have gone wrong.
    root_cause: >-
      variant_audit._StubTower drew its tokens with a seeded CPU generator and
      returned them unmoved. A detector on CUDA would then have mixed host tokens
      with device tensors inside region_embeddings.
    fix: >-
      the draw still uses the same seeded CPU generator — a CUDA generator under
      the same seed draws different numbers — and the result is then moved to
      pixel_values.device. On a CPU host .to("cpu") is a no-op, so every existing
      CPU audit value is bit-identical and no C7 readiness number changes.

  scientific_safety:
    scientific_model_changed: false
    frozen_configs_changed: false
    c3_banks_changed: false
    protocols_or_constants_changed: false
    preflight_disabled_or_weakened: false
    fail_closed_preserved: true      # AUTOGRAD_FAILED, "Stopped BEFORE C4", BLOCKED
    target_access: 0
    c4_to_c13_scientific_status: NOT_RUN
    gemini_calls: 0
    modal_jobs: 0

  tests:
    focused_suite: tests/pipeline/test_gpu_preflight_autograd.py   # 19 passed
    focused_covers:
      - the probe builds a DetectorBatch, not a tensor
      - its composition equals batch_contract_for("G5", ...) exactly
      - a bare tensor is STILL rejected by the detector (the regression itself)
      - the audit stub tower follows the image device, CPU values unchanged
      - forward, scalar finite loss, backward
      - every trainable parameter receives a gradient
      - every gradient is finite
      - a missing gradient is rejected
      - a non-finite gradient is rejected
      - a model with nothing to train is rejected
      - CUDA device verification, including the wrong CUDA index
      - a CPU tensor under a CUDA selection is refused as a silent fallback
      - the probe writes no artifact and resolves no target
    stop_before_c4_test: >-
      tests/pipeline/test_preparation.py::test_an_autograd_failure_stops_before_c4
      — asserts order == [gpu_preflight], EXIT_BLOCKED, "[AUTOGRAD_FAILED]" and
      "Stopped BEFORE C4" on stderr
    pipeline_suite: {passed: 598, failed: 0, skipped: 0}    # tests/pipeline
    detector_suites: {passed: 223, failed: 0, skipped: 1}   # tests/c7, m9, m10 variants
    broad_regression: {passed: 2054, failed: 7, skipped: 101, seconds: 713.14}
    inherited_failure_set_identical: true   # test-id by test-id vs reports/c0/C0_TEST_SUITE.json
    new_unexplained_failures: 0
    tests_weakened_to_obtain_green: 0

  handoff_package: reports/handoff/LINUX_RTX5090_AUTOGRAD_HOTFIX/
  changed_runtime_files:
    - src/prism_fas/pipeline/gpu_preflight.py
    - src/prism_fas/evaluation/variant_audit.py
  full_project_recopy_required: false
  dataset_recopy_required: false
  weights_recopy_required: false
  venv_recopy_required: false
  preferred_deployment: git fetch origin && git checkout <NEW_HEAD>

gpu_deployment_bootstrap_hotfix:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as a deployment/bootstrap engineering hotfix
  discovered_by: >-
    a physical copy of this project running on a second Windows laptop with an
    NVIDIA GPU. Neither defect is reproducible on the build machine by accident;
    both were reported from that host with exact command output.

  defect_1_msys2_host_python:
    symptom: >-
      PATH `python` on the GPU laptop resolved to C:\msys64\mingw64\bin\python.exe.
      A normal Windows CPython 3.12 existed and `py -3.12 train.py` worked, but the
      documented command `python train.py` did not.
    root_cause: >-
      bootstrap.venv_python() chose the environment layout from `os.name == "nt"`
      alone. MSYS2/MinGW Python is a native Windows executable that reports
      os.name == "nt" and sys.platform == "win32" exactly like a standard Windows
      CPython, and then creates a POSIX-scheme environment: .venv/bin/python.exe
      instead of .venv/Scripts/python.exe. `python -m venv` returned 0, so nothing
      detected the mismatch until create_venv looked for an interpreter that was
      never going to be there.
    why_it_was_never_caught_here: >-
      every interpreter on the build machine is a standard Windows CPython, so the
      wrong premise and the right answer coincided.
    fix:
      - explicit host-interpreter classification over measured evidence, returning
        STANDARD_WINDOWS_CPYTHON | MSYS2_MINGW_PYTHON | POSIX_CPYTHON | UNKNOWN_PYTHON
      - >-
        the deciding fact is the scheme `venv` itself uses. Since 3.11 the venv
        module asks sysconfig for get_path("scripts", scheme="venv"); the bootstrap
        asks each candidate interpreter the same question and reads the answer.
        "Scripts" is a standard Windows CPython; "bin" on a Windows-like host is
        MSYS2/MinGW. Path markers, MSYSTEM and the sysconfig build platform are
        recorded as corroborating signals and decide nothing — a standard CPython
        launched from an MSYS2 shell exports MSYSTEM and is still standard, which
        is live in this repository's own Git-Bash tooling.
      - >-
        on Windows, an interpreter that cannot build the environment triggers an
        automatic search for a supported standard CPython through the Python
        Launcher (`py -0p`), with well-known install roots as a fallback. The
        chosen version follows the contract's declared preference, never "newest
        installed" — verified on this machine, where the launcher default is 3.14
        (outside the supported range) and the runner correctly resolved 3.13.
      - >-
        when nothing supported can be found the run stops with
        SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND before any package is installed, and
        prints the detected interpreter, its classification, the supported range
        and every interpreter it discovered.
    msys2_is_not_a_scientific_host: true   # it can create a venv, which is exactly
                                           # why it is dangerous; it is refused

  defect_2_onnxruntime_pin_never_existed:
    symptom: >-
      after `py -3.12 train.py` created .venv\Scripts\python.exe, the install
      failed at onnxruntime==1.24.0 with "no matching distribution", after pip had
      already reached the CUDA index and begun resolving torch 2.13.0.
    root_cause: >-
      onnxruntime 1.24.0 has never been published to PyPI. The 1.24 family begins
      at 1.24.1. The pin arrived with the requirements tree in e5fca49 and was
      never installed anywhere: this project's own .venv contains no onnxruntime at
      all, because the CPU rehearsal profile does not require the science_only
      import group. It was a declared assumption that no run had ever exercised.
    classification: INTENDED_BUT_UNINSTALLABLE_PIN
    selected_pin: "1.24.1"
    selection_rule: >-
      the smallest published patch in the SAME release family, verified against the
      package index rather than assumed. Not latest: 1.24.2/1.24.3/1.24.4 exist and
      were not taken.
    availability_verified: >-
      cp311/cp312/cp313/cp314 wheels for win_amd64, manylinux_2_28 x86_64 and
      aarch64, macOS arm64 — the whole declared Python range on both supported
      platforms.
    declared_in: [requirements/cpu.txt, requirements/cuda-cu126.txt,
                  requirements/cuda-cu129.txt, requirements/constraints.txt,
                  configs/environment/environment_contract.yaml]

  onnxruntime_scientific_safety:
    result_affecting: true    # SCRFDDetector runs the frozen detector through ORT
    verdict: NUMERICALLY_EQUIVALENT_WITHIN_DECLARED_TOLERANCE
    evidence: reports/handoff/ONNXRUNTIME_PIN_EVIDENCE.json
    method: >-
      identical input tensors and identical frozen model bytes
      (weights/face_detectors/scrfd_10g_bnkps.onnx) fed to two isolated
      environments differing only in the ONNX Runtime version. Detections, the
      selected face, its landmarks and the crop geometry were recomputed from each
      runtime's raw outputs by the canonical postprocessing.
    fixtures: 24 CASIA-FASD train frames (12 live, 12 spoof) — SOURCE domain only
    observed: >-
      216/216 output tensors bit-identical; max absolute difference 0.0 on raw
      tensors, scores, boxes and landmarks; 0 detection-count differences; 0
      selected-face differences; 0-pixel crop-geometry difference.
    stated_limitation: >-
      the pin being replaced cannot be instantiated, because it does not exist. The
      reference arm is 1.20.1 — the runtime the frozen Version-B M2 preprocessing
      declared (configs/cloud/modal_m8.yaml, modal_m8.py) — which is the strongest
      historical evidence available. This is a comparison against the last runtime
      this project's preprocessing actually ran under, NOT against the unusable pin,
      and it is recorded as such rather than presented as a direct A/B of 1.24.0.
    target_access: 0
    scientific_impact: >-
      none. No Version-C preprocessing has ever run: data/processed does not exist
      and C4-C13 have never executed scientifically. The pin binds a runtime for
      work that has not started; it cannot invalidate evidence that does not exist.

  defect_3_cuda_profile_with_no_windows_wheel:
    found_by: >-
      auditing the declared CUDA plan against the indices it names, rather than
      assuming it was fine because pip had begun resolving torch. This one had
      not fired yet: it was the next thing that would have gone wrong, and it
      would have gone wrong silently.
    finding: >-
      the cu129 index publishes torch 2.13.0 for manylinux_2_28 x86_64 and
      aarch64 at every supported Python tag, and for NO Windows tag at all.
      cuda-cu129 is the profile a Blackwell or Hopper card selects.
    why_it_would_not_have_failed_loudly: >-
      the CUDA requirement files carry --extra-index-url, so PyPI stays in the
      resolution set. pip would have installed the PyPI torch 2.13.0 win_amd64
      wheel — a different build — while state/ENVIRONMENT_MANIFEST.json went on
      recording profile_id cuda-cu129 and its index. A silent substitution is
      worse than a refusal.
    fix:
      - >-
        the host wheel platform is a selection gate, checked BEFORE compute
        capability, because it is the one criterion that cannot be argued with.
      - >-
        every profile declares the platforms its own index publishes, measured
        rather than assumed. Evidence:
        reports/handoff/CUDA_DEPENDENCY_PLAN_EVIDENCE.json, regenerated by
        scripts/audit_cuda_dependency_plan.py.
      - >-
        a new cuda-cu130 profile carries the SAME torch 2.13.0 / torchvision
        0.28.0 pins from the CUDA 13.0 index, which publishes cp311-cp314
        win_amd64 wheels as well as manylinux.
      - >-
        after installation the bootstrap reads torch.__version__ back out of the
        environment and refuses a build whose local version label is not the
        selected profile's CUDA tag. Reading an index is a claim; this is the
        measurement.
    selection_outcomes:
      windows_blackwell: cuda-cu130      # was cuda-cu129, which has no Windows wheel
      windows_ada_or_ampere: cuda-cu126  # unchanged
      windows_blackwell_driver_below_580: BLOCKED_CUDA_ENVIRONMENT_NOT_VALIDATED
      linux_blackwell_or_hopper: cuda-cu129   # unchanged
      linux_ada_or_ampere: cuda-cu126         # unchanged
    linux_plan_changed: false     # cu129 stays ahead of cu130 in selection order
    torch_version_changed: false  # 2.13.0 / 0.28.0 on every profile, as before
    new_profile: cuda-cu130
    new_profile_status: DECLARED_NOT_VALIDATED_HERE

  venv_recovery:
    why: >-
      the GPU laptop currently holds a .venv that exists, runs and is missing
      packages, because pip died part-way. Telling the operator to delete it by
      hand would break the one-command contract.
    states: [ABSENT, VALID, PARTIAL_NO_INTERPRETER, INTERPRETER_WILL_NOT_RUN,
             WRONG_SCHEME_FOR_HOST, INCOMPATIBLE_PYTHON_VERSION,
             NOT_THIS_PROJECT_VENV, DEPENDENCIES_INCOMPLETE]
    actions: [CREATE, REUSE, INSTALL_INTO, REBUILD]
    dependency_incomplete_is_topped_up_not_rebuilt: true
    manual_deletion_required: false
    rebuild_scope: >-
      only <project>/.venv, only when it carries a virtual-environment layout, and
      never while the interpreter executing the bootstrap is inside it. Three
      guards, because a wrong answer here would delete somebody's system Python.
    self_recreation: REFUSED    # SELF_RECREATION_REFUSED
    validation_after_creation: >-
      exit code 0 from `python -m venv` is not evidence. The interpreter must exist
      at the scheme's path, launch, report a sys.prefix inside the project .venv,
      differ from its base_prefix, and match the host interpreter's Python minor.

  pip_tooling_policy:
    was: unbounded `pip install --upgrade pip` on every install
    observed_on_the_deployment: pip 24.0 -> 26.2.1
    now: BOUNDED_MINIMUM_ONLY
    detail: >-
      pip is upgraded only when it is below the declared floor (24.0) and then only
      within [24.0, 27.0). The deployment's own pip 24.0 already met the floor, so
      the upgrade bought nothing and silently made the resolver that chose this
      project's dependency set a different program from the one it was pinned
      under. setuptools and wheel are never touched. The resolved pip version is
      recorded in state/ENVIRONMENT_MANIFEST.json.

  windows_and_linux_both_supported: true
  layouts: {windows: .venv/Scripts/python.exe, posix: .venv/bin/python}
  hybrid_layout_never_constructed: true    # .venv/bin/python.exe is a defect
  absolute_path_in_scientific_identity: 0

  files_changed_runtime: 8
  files_changed_runtime_list: [train.py, bootstrap.py,
                               configs/environment/environment_contract.yaml,
                               requirements/constraints.txt, requirements/cpu.txt,
                               requirements/cuda-cu126.txt,
                               requirements/cuda-cu129.txt,
                               requirements/cuda-cu130.txt]

  gpu_laptop_hotfix_package:
    path: reports/handoff/GPU_LAPTOP_BOOTSTRAP_HOTFIX/
    runtime_files: 8
    full_project_recopy_required: false
    contains: [HOTFIX_MANIFEST.json, README_APPLY_HOTFIX.md, APPLY_HOTFIX.ps1,
               HOTFIX_INDEPENDENCE_CHECK.json]
    built_by: scripts/build_gpu_bootstrap_hotfix.py
    independence_verified_by: scripts/verify_gpu_hotfix_independence.py
    independence_method: >-
      the pre-fix tree is reconstructed from Git into a temporary fixture with
      sentinel files standing in for the datasets and weights, ONLY the package is
      applied, and the fix is then exercised inside that fixture. A hidden
      dependency on some other changed file would surface as an error rather than
      as an opinion.
    operator_command: python train.py
    manual_py_launcher_required: false
    manual_venv_activation_required: false
    manual_venv_deletion_required: false
    manual_pip_required: false

  tests:
    new_tests_this_milestone: 98
    new_suites: [tests/pipeline/test_bootstrap_host_interpreter.py,
                 tests/pipeline/test_dependency_contract.py,
                 tests/pipeline/test_gpu_hotfix_package.py]
    focused: {passed: 98, failed: 0, skipped: 0}
    pipeline_suite: {passed: 578, failed: 0, skipped: 0}
    broad_regression: {passed: 2034, failed: 7, skipped: 101, seconds: 606.55}
    previous_broad_regression: {passed: 1932, failed: 7, skipped: 101}
    inherited_failure_set_identical: true    # test-id by test-id, not by count
    new_unexplained_failures: 0

  bounded_environment_check_on_this_machine:
    command: python train.py --preflight-only
    path_python_was: C:\Python314\python.exe (3.14.7, outside the supported range)
    resolved_to: C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe
    outcome: PASS   # CPU_FULL_REHEARSAL preflight, environment REUSED, nothing executed
    packages_installed: 0
    network_contacted_by_the_runner: 0
    gpu_allocated: false

  cuda_dependency_plan_audit:
    evidence: reports/handoff/CUDA_DEPENDENCY_PLAN_EVIDENCE.json
    regenerated_by: scripts/audit_cuda_dependency_plan.py
    checked: [supported_python_tags, torch, torchvision, declared_index,
              contract_index, per_platform_wheel_availability]
    gpu_allocated: false          # index queries only; this is plan validation
    per_profile:
      cpu:        {win_amd64: true,  linux_x86_64: true,  linux_aarch64: true}
      cuda-cu130: {win_amd64: true,  linux_x86_64: true,  linux_aarch64: true}
      cuda-cu129: {win_amd64: false, linux_x86_64: true,  linux_aarch64: true}
      cuda-cu126: {win_amd64: true,  linux_x86_64: true,  linux_aarch64: true}
    index_matches_contract: all
    torch_pin_matches_contract: all

  what_did_not_change:
    scientific_protocols_or_constants: 0
    ontology_prompt_schema_route_policy_quotas_selection_rule_seeds: unchanged
    c3_banks_locks_identities: untouched
    search_plans_or_lr_decisions: untouched
    dataset_bytes_or_frozen_weights: untouched
    c4_to_c13_scientific_status: NOT_RUN
    target_access: 0
    gemini_calls: 0
    modal_jobs: 0

# --- physical one-folder asset closure (previous milestone) ------------------
physical_asset_closure:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as asset/path/portability engineering

  why_data_was_empty: >-
    INTENTIONAL EXTERNAL ROOTS, never a deletion. `data/` and `weights/` were
    never tracked by Git and never populated: every large input was resolved
    through absolute paths in the Git-ignored configs/paths.local.yaml, pointing
    at D:\AI on IOT\Anti_spoofing\Dataset and \model_cache. The project-relative
    layout was added later by portable_paths.py, but nothing ever copied the
    bytes in, so resolution silently kept falling back to `paths_local` — and the
    bundle manifest reported present=true from those external paths, which is
    why the folder looked ready while being physically bound to this machine.
  version_b_was_never_a_data_source: true   # the raw roots are a shared external
                                            # Dataset/ tree, not inside Version B

  sources_classified:
    casia_fasd: {class: A_ORIGINAL_RAW, source: "D:/AI on IOT/Anti_spoofing/Dataset/casia-fasd"}
    msu_mfsd:   {class: A_ORIGINAL_RAW, source: "D:/AI on IOT/Anti_spoofing/Dataset/MSU-MFSD"}
    siw_mv2:    {class: A_ORIGINAL_RAW, source: "D:/AI on IOT/Anti_spoofing/Dataset/SiW-Mv2"}
    weights:    {class: A_ORIGINAL_RAW, source: "D:/AI on IOT/Anti_spoofing/model_cache"}
  version_b_artifacts_copied: 0     # policy §4: raw only; no B processed/checkpoint/result

  copied_in:
    casia_fasd: {path: data/raw/casia_fasd, files: 123533, bytes: 2174621978,
                 verdict: IDENTICAL,
                 evidence: "count + total bytes + path/size manifest digest equal;
                            SHA256 of all non-image files and every 50th image
                            (2471 files) — a sampled claim, stated as such"}
    msu_mfsd:   {path: data/raw/msu_mfsd, files: 607, bytes: 11171264770,
                 verdict: IDENTICAL,
                 evidence: "SHA256 of every one of the 606 copied files"}
    siw_mv2:    {path: data/raw/siw_mv2, files: 1702, bytes: 20394222613,
                 verdict: IDENTICAL,
                 evidence: "SHA256 of every one of the 1702 files"}
    weights:    {path: weights/, files: 11, bytes: 2850705424,
                 verdict: IDENTICAL,
                 evidence: "SHA256 source == destination == frozen pin, all 11"}
  copy_method: robocopy physical copy; no symlink, junction or shortcut
  sources_left_intact: true         # COPY only; nothing moved or deleted
  msu_split_archives_excluded: >-
    the MSU source root also holds MSU-MFSD-Publish.zip.001..016 (10.33 GB), which
    are the compressed form of the extracted tree that was copied. The adapter
    reads the extracted tree, so carrying both would duplicate 10 GB for nothing.

  frozen_weight_identity_mismatch: 0
  weights_required_and_present:
    siglip2_frozen_global_tower: {path: weights/pretrained/m9/siglip2, files: 7, stage: C7}
    convnextv2_atto_local_branch: {path: weights/backbones/model.safetensors, stage: C7}
    adaface_identity_backbone: {path: weights/face_identity/pretrained_model/model.pt, stage: C4}
    scrfd_face_detector: {path: weights/face_detectors/scrfd_10g_bnkps.onnx, stage: C6}
    facexformer_parsing: {path: weights/face_geometry/ckpts/model.pt, stage: C6}
  weight_list_derived_from: code and configs, not from the names in the request

  # --- the part copying alone did NOT fix -----------------------------------
  production_fixes: 4
  fixes:
    - id: stale_paths_config_kept_pointing_outside
      was: >-
        ensure_local_paths treated any config naming this project_root as
        authoritative. After the copy it therefore kept resolving the datasets and
        weights on the machine they came from, leaving the folder physically
        self-contained and functionally not.
      now: >-
        a config is also stale when an in-folder copy exists and it points that
        root elsewhere. The in-folder copy wins; a root with NO in-folder copy
        keeps its declared external value, so a machine that legitimately stores
        corpora elsewhere still works; write roots are always inside the project.
    - id: assets_read_the_config_instead_of_the_resolver
      was: >-
        build_assets read model_cache and raw_datasets straight from
        paths.local.yaml. On a fresh copy that file does not exist, so the cache
        was None and all five pinned weights were reported MISSING — on exactly
        the machine the inventory exists to describe.
      now: resolved through portable_paths first, config as fallback
    - id: bundle_preflight_required_what_the_run_produces
      severity: WOULD_HAVE_BLOCKED_THE_GPU_HOST
      was: >-
        bundle_readiness counted data/processed, data/packages, gpat_pairs and the
        evaluation-only label artifact as prerequisites. They are GENERATED_BY_
        PIPELINE, absent on every fresh copy, so `python train.py` on the GPU
        machine would have returned BLOCKED before ever reaching preparation —
        the step that creates them.
      now: >-
        assets whose origin is GENERATED_BY_PIPELINE are reported as
        `produced_by_the_run` and never counted against readiness. Each is still
        gated where it matters: preparation blocks with MISSING_RAW_DATA if it
        cannot build, and every C4-C13 stage re-checks its own inputs.

    - id: scrfd_path_hard_coded_inside_a_hashed_config
      severity: WOULD_HAVE_FAILED_AT_THE_FIRST_PREPROCESSING_STEP
      found_by: the §17 external-reference audit, after the copy
      was: >-
        configs/data/preprocess_m2.yaml line 10 hard-codes
        scrfd_model_path: D:/AI on IOT/.../model_cache/face_detectors/scrfd_10g_bnkps.onnx.
        M2 opens it, so preprocessing on the destination machine would die on a
        path that does not exist there.
      why_the_obvious_fix_was_refused: >-
        that string is inside M2Config.config_hash, which names the work tree and
        is stamped into every M2 manifest row as preprocessing_config_hash.
        Editing the YAML would change a frozen scientific identity, which this
        task does not authorize.
      now: >-
        the declared string is untouched and only the LOOKUP moves:
        M2Config.resolved_scrfd_model_path falls back to the in-folder
        weights/face_detectors/ copy when the declared path is absent. Every site
        that opens the file uses it; nothing that hashes uses it.
      identity_evidence:
        config_hash_before: 8f1e68ef5bc646a24f5b636261c7741c08b79bc9ba46904e3490f111d348c5dd
        config_hash_after: 8f1e68ef5bc646a24f5b636261c7741c08b79bc9ba46904e3490f111d348c5dd
        yaml_bytes_changed: false
        detector_bytes_identical: true      # both copies sha256 5838f7fe...
        matches_frozen_pin: true
      simulated_destination: >-
        with the declared path pointed at a non-existent machine, resolution
        returns the in-folder copy and its sha256 is exactly the frozen pin.
      first_attempt_broke_nine_tests: >-
        the resolver was first added as an M2Config property and the consumers
        were switched to `cfg.resolved_scrfd_model_path`. Several inherited call
        sites pass a duck-typed stand-in config that defines `scrfd_model_path`
        and nothing else, so seven test_full_profile_validation and two
        test_m2_validation_profile_crop_paths cases died with AttributeError.
        Caught by the broad regression, not by the focused suite. The resolver is
        now a module-level `resolve_detector_path(declared)` taking the path
        rather than the config, which works for both shapes; the property remains
        as a convenience. No test was edited to accommodate it.

  # --- verification ---------------------------------------------------------
  active_external_data_dependencies: 0
  active_external_weight_dependencies: 0
  version_b_runtime_dependency: 0
  version_b_independence_evidence: >-
    with checks.VERSION_B_PATH pointed at a non-existent drive, the GPU scientific
    plan resolves ready=True first_stage=C4 with no blockers, and the C0/C1
    adapters pass — they read the committed reports/c0/VERSION_B_INTEGRITY_
    SNAPSHOT.json, not the Version-B repository. The only live reader is
    check_version_b_integrity, which belongs to `--profile validate` alone.
  version_b_caveat: >-
    `python train.py --profile validate` WILL report version_b_integrity FAIL on a
    host without Version B. That is correct and intended — that command audits the
    CLAUDE.md invariant and cannot verify an absent repository — and it is not on
    the normal workflow. Version B was not modified, moved or deleted.

  relocation_test:
    method: same resolver, different absolute root, no .venv, no paths.local.yaml
    raw_and_weights_resolved: in_folder for all four roots
    assets_resolving_outside: 0
    generated_config_roots_inside_project: 10 of 10
    original_D_path_present_in_generated_config: false
  bundle_preflight:
    cpu_rehearsal: {ready: true, required: 16, present: 16}
    gpu_scientific_full: {ready: true, required: 24, present: 24}
    produced_by_the_run: [preprocessed_source_data, source_packages, gpat_pair_plan,
                          target_label_artifact]
    bundle_ready_for_full: "YES"
    blockers: []

  manifests:
    bundle: PORTABLE_BUNDLE_MANIFEST.json    # now records physically_in_folder,
                                             # file_count, size_bytes, identity_relevant
    transfer: PORTABLE_TRANSFER_MANIFEST.json  # NEW; verifies a copied folder
                                               # before training
    transfer_contents: >-
      full SHA256 for 23 identity-critical files (the 11 weight files and the
      frozen configs/locks) and 5 critical files; path+size manifest digests for
      the three datasets, which is a corruption check rather than a byte-identity
      claim and says so.

  sizes_gb:
    casia_fasd: 2.03
    msu_mfsd: 10.4
    siw_mv2: 18.99
    weights: 2.65
    venv: 1.2
    total_folder: 35.32
    recommended_transfer_excluding_venv: 34.12
  safe_to_exclude_from_transfer: [.venv, __pycache__, .pytest_cache,
                                  "data/raw/msu_mfsd/*.zip.0NN split archives"]
  git_bytes_added: 0        # data/raw/ and weights/ are gitignored; 8 files tracked
  license_review: >-
    UNKNOWN, not PROHIBITED. CASIA-FASD, MSU-MFSD and SiW-Mv2 are research
    datasets normally obtained under a signed academic agreement, and no licence
    metadata travels in the trees. This is local machine-to-machine packaging and
    nothing was uploaded anywhere. Whether the destination machine and its
    operator are covered by the existing agreement is the user's call, not a
    technical one — flagged, not decided here.

  tests:
    broad_regression_exact_command: >-
      python -m pytest -q --no-header -p no:cacheprovider
      --continue-on-collection-errors
    broad_regression: {passed: 1932, failed: 7, skipped: 101, seconds: 540.75}
    previous_broad_regression: {passed: 1930, failed: 7, skipped: 101}
    inherited_failure_set_identical: true    # test-id by test-id
    new_unexplained_failures: 0
    skipped_drift: 0
    pipeline_suite: {passed: 476, failed: 0, skipped: 0}
    preparation_suite: {passed: 54, failed: 0, skipped: 0}
    tests_weakened_to_obtain_green: 0
    tests_replaced_because_their_contract_changed: 2

  # --- the post-physical-bundle verification rehearsal (executed) -----------
  post_bundle_rehearsal:
    executed: true
    executed_at_utc: 2026-08-18
    command: "python train.py"          # zero arguments
    verified_at_head: fc58eb2
    exit_code: 0
    wall_clock_seconds: 84
    resolved_intent: CPU_FULL_REHEARSAL
    scientific_eligible: false
    resolved_gpu_scientific_full: false  # correct: no CUDA device on this host
    environment: REUSED
    stages: 14
    substage_modes: 56                  # the corrected figure, reproduced
    all_stages_pass: true
    disk_free_gb_after_the_34gb_copy: 66.59

    # THE point of this run: the assets are now read from inside the folder.
    physical_resolution:
      casia_fasd: {origin: in_folder, root: data/raw/casia_fasd}
      msu_mfsd: {origin: in_folder, root: data/raw/msu_mfsd}
      siw_mv2: {origin: in_folder, root: data/raw/siw_mv2}
      weights: {origin: in_folder, root: weights}
      active_external_data_dependencies: 0
      active_external_weight_dependencies: 0
    paths_local_yaml_rewritten_this_run: false   # it already named in-folder roots,
                                                 # so ensure_local_paths never ran
    configs_namespace_unchanged: true

    preparation_outcome: WOULD_BUILD
    preparation_steps_executed: 0
    derived_trees_created: 0            # data/processed, data/packages,
                                        # gpat_pairs, data/work all still absent
    real_full_data_preparation: NOT_RUN

    raw_data_unmodified: true
    raw_data_evidence: >-
      path+size inventory digest per root, compared before and after, rather than
      a second 34 GB content hash: casia 123533 files / 2174621978 B, msu 607 /
      11171264770, siw 1702 / 20394222613, weights 11 / 2850705424 — all identical.

    c8_sampled_rows: [C-G-RND-P1-s20260806, C-G-RND-P1-s20260807]
    c8_state: {planned: 42, pending: 40, skipped: 2, rows_executed_here: 2}
    c8_rows_reused: true                # both SKIP_VALID_COMPLETE
    c8_matrix_identity_unchanged: true
    counter_drift: 0

    scientific_outputs_modified: 0
    scientific_namespaces_verified_unchanged: [reports/full, runs/full,
                                               assets/recipe_banks, reports/c0,
                                               reports/c1, reports/c2, reports/c3,
                                               configs]
    master_index_rows_claiming_eligibility: 1   # unchanged: the 2026-08-16 C3 run
    writes_confined_to: [reports/rehearsal/, runs/rehearsal/, reports/preflight/,
                         state/]

    target_firewall_with_siw_physically_present:
      target_labels_resolved: 0
      target_metrics_computed: 0
      target_paths_resolved: 0
      target_labels_opened: false
      real_siw_labels_opened: false
      c10_fixture_contains_real_target_data: false
      note: >-
        the whole point of this check. SiW bytes now sit inside data/raw, and the
        rehearsal still never resolves a target label or computes a target metric.

    bundle_preflight:
      cpu_rehearsal: {ready: true, required: 16, present: 16}
      gpu_scientific_full: {ready: true, required: 24, present: 24}
      bundle_ready_for_full: "YES"
    version_b_runtime_dependency_normal_workflow: 0
    version_b_evidence: >-
      with checks.VERSION_B_PATH pointed at an absent drive the normal workflow
      resolves ready with no blockers and C0/C1 pass. `--profile validate` was NOT
      run, per instruction.

    live_provider_calls: 0
    gpu_seconds: 0
    report_html: reports/rehearsal/final/report.html
    report_renders: true
    report_self_contained: true         # 4 embedded figures, 0 external src
    report_leaks_dataset_paths: false
    git_dataset_or_weight_files_visible: 0
    working_tree_after_run: rehearsal and state outputs only

  gpu_seconds: 0
  modal_usage: 0
  gemini_calls: 0
  real_target_scoring: 0
  target_labels_opened: 0
  real_preparation_executed: false     # §25: not run in this task

# --- preparation pipeline coverage closure (previous milestone) --------------
preparation_coverage_closure:
  status: COMPLETE
  branch: portable-one-command-full-run
  is_scientific_execution: false
  authorized_by: user, in session, as a unit/integration test milestone
  scope: tests, plus the minimal production fixes those tests exposed

  gap_that_was_closed: >-
    src/prism_fas/pipeline/preparation.py had zero test coverage and its real
    orchestration had never executed. train.py calls it with
    dry_run=not plan.is_scientific, so every laptop rehearsal took the dry branch
    and the branch that runs FIRST on the collaborator's GPU host had never run
    anywhere.

  new_test_module: tests/pipeline/test_preparation.py
  tests_added: 50
  tests_drive_the_public_entrypoint: true   # preparation.prepare(), not private helpers
  what_is_stubbed: the four canonical builders, and nothing else
  why_stubs_still_prove_something: >-
    each stub creates the tree its real counterpart creates, so the post-build
    completeness check still bites. Step order, dependency chaining, resume
    decisions, validation, failure handling, path resolution and the report are
    all shipping code under test. The stubs mirror the canonical signatures
    exactly, which is how the arity defect below was caught.
  mutation_checked: true
  mutation_result: >-
    reverting either fixed defect fails 32 of the 50 tests, so the suite bites
    rather than merely passing.

  # --- three production defects the tests exposed ---------------------------
  defects_found_and_fixed: 3
  defects:
    - id: finalize_lock_arity
      severity: CRASH_ON_EVERY_REAL_BUILD
      was: "preparation called finalize_lock(root, config, report)"
      truth: "prism_fas.data.package.finalize_lock(package_root, report) takes TWO"
      consequence: >-
        every genuine M3A build died with a TypeError, wrapped as
        PREPARATION_FAILED at m3a_package. `python train.py` on the collaborator's
        machine would have aborted before C4, after the trip was already made.
      fix: dropped the third argument
      guarded_by: test_finalize_lock_is_called_with_the_two_arguments_it_declares
    - id: paths_config_not_portable
      severity: BREAKS_THE_ONE_FOLDER_PROMISE
      was: >-
        every builder was handed configs/paths.local.yaml directly. That file is
        Git-ignored, so a clone has none and load_paths raises FileNotFoundError;
        and a copied folder carries one whose roots still name the machine it left,
        so the builders would write to a D: path that does not exist there.
      fix: >-
        portable_paths.ensure_local_paths(repo) derives a config from THIS folder's
        location when the existing one is absent or names a different project_root,
        and leaves an operator's deliberate config alone when it names this folder.
      verified_non_destructive: true   # this laptop's config hashed identical after
      guarded_by: [test_a_folder_with_no_paths_config_gets_one_describing_itself,
                   test_a_config_naming_another_machine_is_replaced,
                   test_a_config_that_already_names_this_folder_is_left_alone]
    - id: interrupted_tree_looked_complete
      severity: SILENT_CORRUPT_INPUT_TO_C4
      was: >-
        a derived tree counted as present when its directory was non-empty, which
        is equally true of a build that died mid-write. Preparation reported
        NOTHING_TO_DO and C4 would have trained against the half-written package.
      fix: >-
        COMPLETION_MARKERS — a tree is present only once the file its builder
        writes last exists (PACKAGE_LOCK.json for M3A and M3B, PAIR_PLAN_LOCK.json
        for the pair plan).
      guarded_by: [test_a_package_without_its_lock_is_not_reported_as_nothing_to_do,
                   test_a_pair_plan_without_its_lock_is_rebuilt,
                   test_a_finished_tree_is_still_nothing_to_do]

  # --- what the tests establish --------------------------------------------
  verified_step_order: [m2_preprocess, m3a_package, m3b_priors, gpat_pairs]
  verified_step_order_source: read from preparation.STEPS, not from prose
  builders_delegated_to:
    m2_preprocess: prism_fas.data.m2_runner.run
    m3a_package: prism_fas.data.package.{load_package_config,build_package,
                                         validate_package,finalize_lock}
    m3b_priors: prism_fas.data.package.m3b.build_m3b_package
    gpat_pairs: prism_fas.synthesis.pair_plan.{write_pair_plan,pair_plan_identity}
  scientific_semantics_unchanged: true
  no_scientific_constant_contract_or_search_plan_changed: true
  corpus_truncation_guard: >-
    m2_runner.run defaults limit_records=3, a smoke default that would silently
    truncate the scientific corpus. A test asserts the real record count reaches
    the builder and that the value is never 3.

  resume_scenarios_proven: [A_fresh_build, B_all_valid_nothing_to_do,
                            C_partial_reuses_what_is_valid, D_interrupted_rebuilds,
                            E_stale_fails_closed, F_failed_builder_raises]
  rerun_report_identical_except_timing: true
  failure_atomicity: >-
    a failing step raises PreparationError naming the step and the completed ones;
    later steps do not run, nothing is marked complete, and a rerun resumes from
    the failed step rather than rebuilding what succeeded.
  no_silent_success: >-
    a builder that returns success without producing its tree is caught by the
    post-loop re-check and raises with the tree named.

  relocation_verified: true
  relocation_method: the same fixture built under two different absolute roots
  identities_survive_relocation: true
  absolute_host_paths_identity_relevant: false

  target_firewall:
    preparation_needs_target_labels: false
    target_dataset_preprocessed: false          # siw_mv2 is never passed to m2
    absent_target_blocks_preparation: false
    reads_inside_the_target_raw_root: 0         # asserted by a patched Path.open
    real_siw_files_used_in_tests: 0

  fixture_outputs_can_be_scientific_ancestors: false
  ancestry_check: >-
    every C4-C13 inherited input still resolves under reports/full or runs/full, and
    none of them names anything preparation writes.

  tests:
    preparation_suite_exact_command: >-
      python -m pytest tests/pipeline/test_preparation.py -q --no-header
      -p no:cacheprovider
    preparation_suite: {passed: 50, failed: 0, skipped: 0}
    focused_groups_exact_command: >-
      python -m pytest tests/pipeline/test_preparation.py
      tests/pipeline/test_portable_runner.py
      tests/pipeline/test_full_path_and_ancestry.py -q --no-header -p no:cacheprovider
    focused_groups: {passed: 145, failed: 0, skipped: 0}
    pipeline_suite: {passed: 471, failed: 0, skipped: 0}
    broad_regression_exact_command: >-
      python -m pytest -q --no-header -p no:cacheprovider
      --continue-on-collection-errors
    broad_regression: {passed: 1927, failed: 7, skipped: 101, seconds: 510.57}
    previous_broad_regression: {passed: 1877, failed: 7, skipped: 101}
    net_new_passing_tests: 50
    inherited_failure_set_identical: true   # test-id by test-id
    new_unexplained_failures: 0
    documented_failures_now_passing: 0
    skipped_drift: 0
    tests_weakened_to_obtain_green: 0

  # --- the post-fix verification rehearsal (executed, not inferred) ---------
  post_fix_rehearsal:
    executed: true
    executed_at_utc: 2026-08-17T16:00:31Z
    command: "python train.py"          # zero arguments, the normal user workflow
    run_id: rehearsal-20260817T160031Z-419e83f9
    verified_at_head: 02e56cc
    exit_code: 0
    wall_clock_seconds: 92
    resolved_intent: CPU_FULL_REHEARSAL
    scientific_eligible: false
    environment: REUSED                 # profile=cpu VALIDATED, id 7212f5ca5fc35dd9
    gpu_detected: none                  # nvidia-smi not on PATH
    stages: 14
    substage_modes: 56
    all_stages_pass: true
    checks_failed: 0

    preparation_mode: WOULD_BUILD       # the safe dry branch
    preparation_steps_executed: 0
    preparation_paths_config_action: null   # ensure_local_paths never invoked
    real_full_data_preparation: NOT_RUN
    configs_namespace_unchanged: true   # hashed before and after; proves the dry
                                        # branch never wrote a paths config

    c8_sampled_rows: [C-G-RND-P1-s20260806, C-G-RND-P1-s20260807]
    c8_planned: 42
    c8_skipped: 2                       # held at 2; did not climb between runs
    c8_rows_reused_from_stored_artifacts: 2
    c8_matrix_identity_unchanged: true  # a777671f...
    c8_complexity_table_identical: true # byte-identical to the accepted run
    counter_drift: 0

    scientific_outputs_modified: 0
    verified_by: sha256 over reports/full, runs/full, assets/recipe_banks, configs,
                 reports/c0, reports/c1, reports/c2, reports/c3 — all UNCHANGED
    writes_confined_to: [reports/rehearsal/, runs/rehearsal/, reports/preflight/,
                         state/MASTER_RUN_INDEX.json, state/PIPELINE_STATE.json]
    master_index_rows_claiming_eligibility: 1   # unchanged: the 2026-08-16 C3 live run
    rows_added_by_this_run_claiming_eligibility: 0

    live_provider_calls: 0
    gpu_seconds: 0
    modal_usage: 0
    gemini_calls: 0
    real_target_access: 0               # target_labels_resolved=0,
                                        # target_metrics_computed=0,
                                        # target_paths_resolved=0, all fixture_backed
    c4_to_c13_scientific_status: NOT_RUN

    report_html: reports/rehearsal/final/report.html
    report_renders: true
    report_self_contained: true         # 4 embedded data-URI figures, 0 external src
    report_declares_itself: "REHEARSAL - NOT SCIENTIFIC EVIDENCE"
    linked_tables_present: 9
    one_command_runner_broken_by_the_fixes: false

  coverage:
    tool: stdlib sys.settrace   # the repository declares no coverage dependency
    preparation_py: {covered: 207, executable: 215, percent: 96.3}
    untested_lines: 8
    untested_region: _record_count body
    untested_reason: >-
      it delegates to the canonical dataset adapter over a real dataset root, which
      a fixture has no copy of. It has no branches, so its only failure mode is a
      moved signature — which is asserted directly against load_paths, adapter_for
      and DatasetDefinition instead.
    every_result_affecting_branch_tested: true

execution_adapters:
  restructure_plan_step: 4 of 6 (partial — C0-C3 only)
  authorized_by: user, in session, engineering-only adapter milestone
  package: src/prism_fas/pipeline/adapters/
  modules: [__init__, context, historical, c3, c3_live, quota, fixtures, registry]
  design_rule: >-
    adapters are THIN. Every one delegates to the canonical implementation —
    RouteContext for the contract and request, RecipePlanner for retry and
    validation, arm_schedules for RND/DET, recipes.selection for the selector,
    pipeline.checks for identity verification. No scientific logic is duplicated.
  c0_c1_c2_c2b_c2c: VERIFICATION_ONLY   # provider binding is NONE and cannot change
  historical_provider_calls_repeated: 0
  c3_modes: [PRE_LIVE_VERIFY, LIVE_GENERATE, RESUME_LIVE_GENERATE, FINALIZE_BANKS]
  c4_to_c13: IMPLEMENTED_ENGINEERING_ONLY   # 2026-08-17; see c4_to_c13_engineering_readiness
  c4_to_c13_provider_binding: NONE          # no stage from C4 onward can reach a provider

  provider_binding_model: >-
    mode (what the adapter does) is separate from binding (what it may talk to).
    That separation is what lets smoke drive the entire live control path against
    fixtures while remaining structurally unable to reach Gemini.
  live_binding_requires_all_of:
    - the profile's compute policy permits a live provider   # full only
    - the stage is in live_provider_permitted_for_stages     # C3 only
    - explicit human authorization for the run               # CLI flag
    - a materialized quota snapshot
    - a present GEMINI_API_KEY
  live_binding_is_never_a_default: true   # omitting the choice yields MOCK, not LIVE

  validate_c0_c3:
    command: python train.py --profile validate --from C0 --to C3
    outcome: PASS
    checks_run: 11
    checks_failed: 0
    provider_calls: 0
  smoke_c0_c3:
    command: python train.py --profile smoke --from C0 --to C3
    outcome: PASS
    substages_executed: [C0, C1, C2, C2B, C2C, C3]
    engineering_status: SMOKE_PASS        # scope: stages C0..C3 ONLY
    scientific_status: NOT_RUN
    provider_calls_live: 0
    provider_calls_mock: 12               # the C3 fixture rehearsal
    rehearsal_namespace: reports/smoke/c3/live/
    scientific_namespace_untouched: true  # verified by hashing reports/c3/live/ before
                                          # and after; the offline profiles change nothing

  c3_live_state_machine:
    path_when_scientific: reports/c3/live/C3_LIVE_GENERATION_STATE.json   # now populated
    statuses: [NOT_STARTED, IN_PROGRESS, COMPLETED_VALID, FAILED_RETRYABLE,
               FAILED_BLOCKING]
    completed_valid_is_terminal: true     # never re-issued by restart or --resume
    drifted_archive_action: FAIL_CLOSED   # never silently regenerated
    scenarios_proven_offline: [A_crash_resume, B_retryable_same_request,
                               C_quota_block_preserves, D_resume_continues,
                               E_zero_calls_when_complete, F_corrupt_fails_closed]

execution_layer:
  restructure_plan_step: 2 of 6   # docs/V15_PIPELINE_RESTRUCTURE_PLAN.md §4
  authorized_by: user, in session, in place of the recorded next_authorized_action
  entrypoint: train.py
  package: src/prism_fas/pipeline/
  modules: [profiles, status, stages, state, registry, resume, budget, checks,
            orchestrator]
  profile_configs: configs/execution/{validate,smoke,full}.yaml
  profile_identities:
    validate: b1ab119c7552ec852a6f44d8254eb791549ae767042ea6a773cd8dab975edd31
  validate_run:
    outcome: PASS
    stages_traversed: 14
    stages_with_real_checks: 4      # C0, C1, C2, C3
    stages_not_applicable: 10       # C4..C13 — no checks exist to run
    checks_run: 11
    checks_failed: 0
    evidence: reports/validate/
    engineering_status_axis: NOT_TESTED for every stage
    why_not_smoke_pass: >-
      L.3 fixes the engineering vocabulary at NOT_TESTED | RUNNING | SMOKE_PASS |
      SMOKE_FAIL | BLOCKED. None of those means "validate passed". Borrowing
      SMOKE_PASS would claim a smoke execution that never happened and inventing a
      sixth value would edit a frozen vocabulary, so the run reports on a separate
      validate_gate axis and leaves engineering_status at NOT_TESTED.
  scientific_eligibility_rule: >-
    an artifact is scientifically eligible only when the profile permits it AND the
    run outcome is PASS. A BLOCKED full run permits eligibility and earns none,
    because L.2 validates eligibility from the ancestor identity chain and a blocked
    run has no ancestry. Zero rows in MASTER_RUN_INDEX.json claim eligibility.

historical_milestones:
  C0: ACCEPTED_WITH_DOCUMENTED_DEVIATION   # froze against v1.1; v1.5 execution layer is new scope
  C1: ACCEPTED_WITH_DOCUMENTED_DEVIATION   # schema-identity naming corrected prospectively at C2C
  C2: ACCEPTED                             # disposable pilot; never enters C3
  C2B: ACCEPTED_WITH_DOCUMENTED_DEVIATION  # outcome BATCH_SHAPE_FAIL, preserved unchanged
  C2C: ACCEPTED                            # route contract exactly ["physics","gpat"]
  C3_preparation: COMPLETE                 # generation contract + selection contract, both bound
  C4..C13: ENGINEERING_READY_SCIENTIFICALLY_NOT_RUN

c3:
  selection_implementation: COMPLETE        # prism_c3_selection_v1, §7.8.3
  contract_supersession: COMPLETE           # §7.8.4 superseding pre-scientific lock written
  scientific_generation: COMPLETE           # executed live 2026-08-16, all 12 requests valid

  generation_contract_identity: 884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85
  selection_contract_identity: 3d4675ba16b39d10f0e888f3c523ea540647544a436ff387bc84f2c17eced070
  bank_contract_identity: d6105f8de601ae94cb0d46e087a0ebe664b3b5df9d193f5797e306bfe4fe03b8
  rnd_schedule_identity: 0e21f96a55bdb1c71f1b49dee39136effb3c3eaffae0d74a5b7557c333033aeb
  det_schedule_identity: bed9a3ab5e25fa3ca09f515479839856d4ef37db62d650a55abfea9c89cb0ae2

  preliminary_bank_lock_identity: 7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba
  preliminary_bank_lock_file_sha256: a755f88058a7ebbd60dd784444b3319baefd9d163d1fd970e47335525578eab8
  preliminary_bank_lock_bytes_unchanged: true

  superseding_bank_lock_identity: 1acdf68f56195f1b568449b545865ae2868d99d480ed6b75b28215178c5e9628
  superseding_bank_lock_path: reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json
  superseding_bank_lock_status: PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK
  supersession_reason: >-
    SELECTION_CONTRACT_WAS_REQUIRED_BY_GOVERNING_SPEC_BUT_NOT_IDENTITY_BOUND_BEFORE_C3_GENERATION
  scientific_requests_before_supersession: 0

  c3_scientific_logical_requests: 12
  c3_scientific_candidate_slots: 384
  raw_candidate_archives_created: 12        # reports/c3/live/raw_responses/
  recipe_banks_selected: 3                  # LLM, RND, DET
  live_wiring: EXECUTED
  live_provider_attempts: 13                # 12 logical requests + 1 retry of req-03
  status: SCIENTIFIC_BANKS_FROZEN

  scientific_execution:
    executed_at_utc: 2026-08-16
    profile: full
    provider_binding: live
    model_id: gemini-3.6-flash              # provider-returned model_version matched
    logical_requests: 12
    objects_per_request: 32
    llm_raw_slots: 384
    provider_attempts: 13
    extra_logical_requests: 0
    retried_requests: [c3-llm-req-03]       # 2 attempts, same logical request (§7.6)
    rate_limit_incidents: 0
    quota_blocks: 0
    pacing_seconds_between_requests: 15     # transport only; request bytes untouched
    quota_snapshot_sha256: 763674ea101e2ee3b97bea69405d57e567e78afdb8a56515ea322829561b9efe

  arms:
    LLM: {raw: 384, eligible: 384, selected: 256,
          bank_identity: f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59}
    RND: {raw: 384, eligible: 384, selected: 256,
          bank_identity: 07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136}
    DET: {raw: 384, eligible: 384, selected: 256,
          bank_identity: 2802ca5f537c4278eefdb160049d52cb1b667234ec5e32736a733b272e9231c9}
    rejections_at_any_eligibility_stage: 0
    cross_arm_content_overlap: 0
    all_selected_routes: [physics, gpat]

  scientific_bank_lock:
    path: reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json
    identity: eb57cdfcab8fce1640a5e639e1234635409bc946938419c34020830e641a845d
    banks: assets/recipe_banks/c3/{llm,rnd,det}/
    supersedes_but_preserves: [7ee96d3a... preliminary, 1acdf68f... pre-scientific]

  pre_live_audit:
    gate: PASS
    evidence: reports/c3/v15_pre_live_audit/
    verified_at_utc: 2026-08-14
    verified_commit: 8763dc986861b5b11e45297863ef5bc2bff5b3c0
    verified_worktree_clean: true   # the e850e99 run stamped its parent while the new
                                    # C3 tests were still untracked; this run is clean
    requirements_mapped: 25
    requirements_passing: 25
    provider_call_delta: 0
    provider_fingerprint: de3ba4703be8cd23c73b1d49fc7db01952f6ad81a4cb59c415f48e85d99bd273
    provider_fingerprint_scope: PROVIDER_EVIDENCE_ALLOWLIST   # 75 files; excludes this
                                                              # audit's own output
    provider_fingerprint_identical_across_three_full_runs: true
    verified_preflight_worktree_dirty_paths: []
    identities_reproduced: [generation, selection, bank_contract, superseding_lock,
                            rnd_schedule, det_schedule]
    meaning: >-
      Verification evidence only. PASS means the frozen contract is ready for USER REVIEW.
      It does NOT authorize live Gemini generation.

  selection_contract_summary:
    version: prism_c3_selection_v1
    raw_candidate_slots_per_arm: 384          # 12 logical requests x exactly 32 objects
    minimum_eligible_pool_per_arm: 320        # below this, C3 FAILS for that arm
    final_bank_size_per_arm: 256
    arms: [RND, DET, LLM]
    required_generator_route: [physics, gpat]
    objective_order: [hard_feasibility, preferred_shortfall, single_value_balance,
                      multi_label_balance, canonical_tie_break]

execution:
  current_profile: smoke
  engineering_status: SMOKE_PASS   # the EXECUTION LAYER over C0-C13
  scientific_status: NOT_RUN       # for C4-C13; C3's own scientific status is above
  # These now exist and were written by a real `python train.py --profile validate`.
  # They are navigation aids under L.10, never scientific authority.
  pipeline_state: {path: state/PIPELINE_STATE.json, exists: true}
  master_run_index: {path: state/MASTER_RUN_INDEX.json, exists: true, run_count: 14,
                     rows_claiming_scientific_eligibility: 0}
  orchestrator: {path: train.py, exists: true}

source_search:
  active: false
  search_space_identity: N/A
  configs_tested: []
  configs_remaining: []
  current_source_winner: N/A
  selected_config_lock: N/A

contract_identities:
  ontology: 90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd
  system_prompt: e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737
  batch_generation_template: e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20
  coverage_quota: 89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5
  single_recipe_schema: 1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579
  batch_envelope_schema: f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113
  route_policy: 209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7
  provider_config: 3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da

active_locks:
  - "path: reports/c3/C3_BANK_LOCK.json identity: 7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba status: FROZEN, immutable, SUPERSEDED — historical evidence, bytes preserved unchanged"
  - "path: reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json identity: 1acdf68f56195f1b568449b545865ae2868d99d480ed6b75b28215178c5e9628 status: PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK — AWAITING USER APPROVAL note: >- A CONTRACT lock. It does not assert that any per-arm 384-slot raw archive or 256-recipe bank exists; none has been generated or selected."

historical_live_provider_calls:
  total_attempts_before_c3: 47      # 46 archived + 1 documented-but-overwritten (C2B 400)
  semantic_responses: 36
  c1: 0
  c2_smoke: 2
  c2_pilot: 42
  c2b: 1 archived (+1 documented in C2B_ENVELOPE_REJECTION.json)
  c2c: 1
  c3_scientific: 13     # the authorized live run on 2026-08-16: 12 requests + 1 retry
  c4_to_c13: 0          # no stage from C4 onward can construct a provider at all

tests:
  latest_exact_command: python -m pytest -q --no-header -p no:cacheprovider --continue-on-collection-errors
  passed: 1932
  failed: 7
  skipped: 101
  measured_at_utc: 2026-08-18   # after the physical one-folder asset closure
  milestone_suites: {C0: 32, C1: 138, C2: 43, C2B: 41, C2C: 54, C3: 156, C7: 19,
                     pipeline: 313}
  new_tests_this_milestone: 97   # 26 search engine + 35 portability/contracts +
                                 # 19 decision contract + 17 pipeline
  pipeline_suite_exact_command: python -m pytest tests/pipeline -q --no-header -p no:cacheprovider
  pipeline_suite: {passed: 476, failed: 0, skipped: 0}
  c7_suite_exact_command: python -m pytest tests/c7 -q --no-header -p no:cacheprovider
  c7_suite: {passed: 19, failed: 0, skipped: 0}
  pipeline_offline: >-
    sockets blocked and ambient credentials deleted by an autouse fixture. The static
    AST check still forbids a provider, Modal, torch or target-evaluation import, with
    a short allowlist of LAZY exceptions: C4/C5/C7/C8 import torch inside functions to
    prove instantiate/forward/backward on a CPU fixture, and C11 imports the canonical
    prediction builder rather than duplicating it. The guarantee is enforced at
    runtime instead of by the string rule: a subprocess probe imports every pipeline
    and adapter module and asserts that no torch, Modal or vendor SDK was loaded.
  inherited_known_failures: 7   # exactly the set documented in reports/c0/C0_TEST_SUITE.json
  inherited_failure_set_identical: true   # compared test-id by test-id, not by count
  new_unexplained_failures: 0
  c3_suite_exact_command: python -m pytest tests/c3 -q --no-header -p no:cacheprovider
  c3_suite: {passed: 156, failed: 0, skipped: 0}
  c3_offline: sockets blocked, ambient credentials deleted, no provider SDK on the C3 path

known_deviations:
  - C1 recorded 7afc3abd… as a single-recipe schema identity; it is the 32-object envelope
    identity. Corrected prospectively in docs/c2c/C2C_IDENTITY_CORRECTION_NOTE.md.
  - The C1-recorded bounded 32-object envelope is rejected by the provider (400
    INVALID_ARGUMENT); the working envelope omits the array bound and enforces exactly-32
    on the response.
  - C2B outcome BATCH_SHAPE_FAIL is preserved and must not be rewritten.
  - Spec copied to docs/ per the bootstrap prompt; §M.1 canonical layout says docs/specs/.

blockers:
  - "C4-C13 are ENGINEERING_READY and SCIENTIFICALLY NOT_RUN. Every one of them BLOCKS
    under --profile full on this machine and names the input it lacks. As of this
    milestone the reason is exclusively DATA and HARDWARE: the scientific code path
    exists for all ten stages and contains no placeholder. The blocking inputs are
    enumerated per stage in final_full_path_closure.c4_to_c13_full_path."
  - "CUDA HARDWARE VALIDATION = NOT YET EXECUTED. The runner grades a host generically
    on compute capability and driver version, and the classifier was exercised on ten
    synthetic hosts — but no real CUDA device has ever been allocated by this project.
    cuda-cu129 and cuda-cu126 are DECLARED_NOT_VALIDATED_HERE. The first real GPU run
    is therefore also the first validation of the wheel selection and of the pre-C4
    GPU preflight."
  - "The production FULL output audit is STRUCTURAL, not observed. No
    reports/full/c4..c13 artifact set has ever existed. The audit is sound only because
    the writers are the same code under both contexts; it is not a record of a full run."
  - "RESOLVED — was: preparation.py has ZERO test coverage and its real build path has
    never executed. Closed by preparation_coverage_closure below: 50 focused tests now
    drive the real `prepare()` entrypoint with the four canonical builders stubbed, and
    THREE production defects that were hiding behind the gap are fixed. What remains is
    narrower and is stated as its own blocker: preparation has still never run against
    the real corpora."
  - "Preparation has never run against REAL data. The orchestration, ordering, resume,
    validation, failure handling and path resolution are exercised; what the stubs stand
    in for — the actual m2/m3a/m3b/pair-plan builds over CASIA and MSU — has not run
    here and cannot without the raw corpora and hours of compute. Those builders are
    themselves inherited and separately tested, so the untested surface is their
    interaction with real data volume, not their logic."
  - "CORRECTED by the pre-GPU audit. The previous milestone listed the pinned weights and
    AdaFace as missing; they are PRESENT and now verified by hash against their frozen
    pins — SigLIP2 (all 7 files), ConvNeXt V2 Atto, AdaFace ir50, SCRFD and FaceXFormer,
    ~2.8 GB in the declared model cache. That inventory looked under
    data/packages/pretrained and hard-coded present=false for AdaFace instead of
    resolving the cache; the resolver is fixed. The genuine C4 gap is DERIVED data:
    data/processed, data/packages and data/packages/gpat_pairs are absent, plus a real
    GPU. The raw CASIA/MSU/SiW roots are present, so the derived trees are rebuildable
    on the GPU host."
  - "Missing scientific inputs for C10-C12 — the SiW-Mv2 v2 feature package under its
    declared read-only policy, and the evaluation-only label artifact for the isolated
    C-G8 scorer. Neither was resolved, opened or hashed by this milestone."
  - "NEEDS_SCIENTIFIC_DECISION before C4 or C7 Track R may execute under --profile full.
    The forensic audit is now COMPLETE and the options are costed in
    reports/handoff/LR_ANCHOR_DECISION_DOSSIER.json — three legal interpretations for
    C4 and Track R, with B (a single coordinate scaling every active LR group and
    preserving inherited ratios) recommended. Track G needed no decision: backbone_lr
    controls zero parameters there, so head_lr is already the unique anchor. Approving
    an interpretation changes both frozen search-plan identities, which is why it must
    land before the external-GPU plans are frozen."
  - "Steps 3, 5 and 6 of the restructure plan remain — dual-status stamping of newly
    written artifacts beyond the pipeline's own, the Appendix M.1 layout move with
    identity re-derivation, and moving the stage adapters into
    src/prism_fas/pipeline/stages/."
  - "Quota remains NOT_PROGRAMMATICALLY_AVAILABLE for this tier. The manual AI Studio
    step is now DONE — reports/c3/live/C3_QUOTA_SNAPSHOT.json records the user's
    observation with current_remaining_rpd=UNKNOWN. A future live run needs a fresh
    observation."

compute_policy:
  # Operational only. L.12: compute budget is never a treatment factor and never enters
  # a scientific identity.
  modal_budget_remaining_usd_approx: 30
  modal_budget_authorized_for_full_scientific_training: false
  modal_gpu_spent_this_milestone: 0
  intended_strategy: >-
    PHASE A (done): complete C0-C13 engineering readiness on local CPU, mocks, fixtures
    and tiny smoke budgets. PHASE B (later): execute the full scientific pipeline on a
    separate, sufficiently resourced GPU supplied by the user's collaborator. The
    remaining Modal credit is preserved rather than spent on discovering engineering
    defects.
  backend_neutrality_verified: true   # a scientific identity is invariant across every
                                      # declared backend; proven by construction, not asserted

deviations_recorded_this_milestone:
  - "check_c3_generation_not_started was RETIRED and replaced by
    check_c3_scientific_banks_frozen. The old check asserted that no C3 generation
    evidence existed — true when written, false since the authorized live run completed
    on 2026-08-16 — and it kept passing only because its globs pointed at
    reports/c3/raw_responses/ while the archives were written to
    reports/c3/live/raw_responses/. A check that asserts a false thing and passes by
    accident is worse than no check, so the obligation was moved rather than deleted:
    validate now proves the frozen banks are complete and re-derive. No scientific
    artifact changed."
  - "A new detector fusion value `glr_concat` was ADDED to implement the v1.5 §13.4.2
    Track-R decision contract, which the inherited detector could not express: its
    `prism_noisy_or` path fuses post-hoc SCORES and its `simple_concat` path omits the
    9-region summary entirely, so no inherited variant satisfies §13.5's requirement
    that fused_logit_R depend directly on g, l and r. The change is strictly additive.
    Every inherited variant keeps its exact architecture identity — the decision-graph
    fields enter the payload only for the new fusion — so the frozen M9 reference
    checkpoint stays loadable, m10_matrix_identity is unchanged at a4972b0d..., and no
    existing matrix row uses the new value. The vocabulary entry was added to
    configs/experiments/m10_matrix.yaml because a test asserts the config and the code
    accept the same flag values."
  - "The pipeline import guard was relaxed from `no torch or target-evaluation import
    anywhere` to `no such import except a named lazy one`. C4/C5/C7/C8 must run torch to
    prove instantiate/forward/backward on a CPU fixture, and C11 must use the canonical
    prediction builder rather than a second implementation. Each exception is listed
    individually, each must stay inside a function, and the guarantee the rule was
    protecting is now enforced by measurement: a subprocess probe imports every pipeline
    and adapter module and asserts no torch, Modal or vendor SDK was loaded."
  - "A real engineering defect was found by the second smoke run and fixed:
    `--profile smoke --resume` resolved C3 to RESUME_LIVE_GENERATE and routed it through
    a bare request, leaving the mock binding with no scripted provider and BLOCKING the
    run. A generating mode under smoke is now always given its fixtures. This is exactly
    the class of defect the smoke profile exists to surface before GPU time is spent."
  - "Test sandboxes now copy assets/. C3's bank check and C5's route check read the
    frozen recipe banks, so a sandbox without them is corrupt rather than minimal."

deviations_recorded_in_the_previous_session:
  - The v1.5 execution layer was built ahead of the recorded next_authorized_action. The
    user authorized it explicitly in session; the C3 review it displaced is unchanged and
    is restored as the next authorized action below. No C3 artifact, lock or identity was
    touched, and the validate run re-derived all of them unchanged.
  - "L.3 provides no engineering-status value meaning 'validate passed'. Recorded as an interpretation rather than silently reconciled: engineering_status stays NOT_TESTED and the run reports on a separate validate_gate axis. See execution_layer above."
  - .gitignore re-includes reports/{validate,smoke,full}/ so execution evidence is
    committable, matching the existing rationale for the reports/c0..c13 roots.
  - C2B and C2C are modelled as SUBSTAGES of C2 rather than stages of their own. L.9
    fixes the milestone sequence at C0..C13; inventing a C2B stage would change a
    numbering the spec owns. Each substage still gets its own evidence row.
  - A non-scientific profile writes its C3 live state under its own reports namespace
    (reports/smoke/c3/live/) rather than reports/c3/live/, so a fixture rehearsal can
    never be mistaken for scientific generation evidence.

# ==========================================================================
# C7-C13 PRODUCTION-PATH READINESS AND THE C7 SOURCE-SEARCH ARM DECISION
# ==========================================================================
#
# What this milestone did NOT do: run C7 or C8 scientifically. No detector was
# trained, no scientific metric exists, and no stage below reports a scientific
# PASS. What it did is make the scientific paths REAL and fail-closed, and find
# - on a laptop, before a GPU hour was spent - the class of wiring defect the
# three C6 reruns cost.
c7_c13_readiness_milestone:
  milestone: C7_C13_PRODUCTION_PATH_READINESS
  executed_on: 2026-08-24
  execution_profile: none   # audit + rehearsal + offline tests only
  scientific_execution: NONE
  target_access: 0
  gpu_used: false

  # --- the scientific decision this milestone froze -------------------------
  c7_source_search_synthetic_arm:
    decision_id: C7_SOURCE_SEARCH_SYNTHETIC_ARM
    value: DET
    status: FROZEN
    source: EXPLICIT_SCIENTIFIC_DECISION
    spec_status: UNDER_SPECIFIED_IN_V1_5
    timing: BEFORE_FIRST_C7_SCIENTIFIC_TRIAL
    frozen_on: 2026-08-24
    record: configs/search/c7_source_search_decision.yaml
    decision_identity: ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
    loader: prism_fas.search.c7_decision.load_decision
    question: >-
      15.2.2 freezes the bounded detector/loss SOURCE_SEARCH envelope completely
      - coordinate order, multipliers, one pass, ranking tuple, tie-break - and
      never names which of C6's three matched banks supplies the synthetic
      quarter of the batch while the search runs. Result-affecting, so it could
      not be guessed.
    rationale:
      - >-
        DET is the structured NON-LLM control. A generator arm, exercising the
        same synthetic machinery, but not the thing under test.
      - >-
        LLM is the proposed treatment. Tuning the common detector configuration
        on the treatment bank and then comparing that treatment against its own
        controls would be a tuning advantage no statistic removes.
      - >-
        DET is the only generator arm PRIMARY in both tracks. Track G runs
        RND/DET/LLM and Track R runs DET/LLM, so one non-treatment anchor serves
        both track searches. Track-R RND is a compute-conditional diagnostic,
        not a primary required row.
      - >-
        Conservative for the LLM claim. The treatment receives no
        treatment-specific hyperparameter tuning.
      - >-
        Frozen before any C7 scientific metric existed, so it cannot have been
        chosen from a result.
    not_a_spec_claim: >-
      v1.5 does not choose DET. This is an explicit closure decision over an
      under-specified input, recorded as such.
    prohibited_after_freeze:
      - RND
      - LLM
      - a separate hyperparameter search per generator arm
      - pooled or result-dependent arm selection
      - a second pass that changes the arm
      - >-
        a new pooled RND/DET/LLM search bank, an arm-balanced sampler, a rotated
        batch composition, an averaged arm objective or per-arm hyperparameter
        voting
    reopening_policy: >-
      config_G and config_R are immutable for the C8 primary comparisons. A poor
      RND, DET, LLM, P1, P2 or P3 result does not reopen C7. No second pass, no
      arm-specific rescue search, no LLM-specific fine-tuning.

  # --- what C7 now is -------------------------------------------------------
  c7:
    engineering_readiness: PASS   # unchanged CPU fixture obligation
    scientific_executor: IMPLEMENTED
    scientific_status: NOT_RUN
    workflow_dispatch: >-
      _engineering_workflow / _scientific_workflow, chosen on the context
    requires_gpu_for_science: true
    searches: 2   # one bounded pass per TRACK, never per arm
    track_g:
      search_population: C6 DET matched bank
      variant: TRACK_G_FLAGS
      tunes: [learning_rate_multiplier, weight_decay, warmup, lambda_syn,
              lambda_risk]
      not_applicable: [lambda_local, lambda_MIL, lambda_P, lambda_M, lambda_out,
                       lambda_clean]
      why: >-
        Track G instantiates no ConvNeXt, no regions, no manifold and no
        PromptHead, so those terms never evaluate. 15.2.2 skips inactive terms
        rather than inventing a weight for them.
      c8_consumers: [C-G-RND, C-G-DET, C-G-LLM]
    track_r:
      search_population: C6 DET matched bank
      variant: TRACK_R_FLAGS   # manifold OFF
      tunes: [learning_rate_multiplier, weight_decay, warmup, lambda_syn,
              lambda_local, lambda_MIL, lambda_P, lambda_risk]
      not_applicable: [lambda_M, lambda_out, lambda_clean]
      why: >-
        manifold=OFF primary Track R executes no L_real / L_out / L_clean, so the
        K=4-only weights are NOT_APPLICABLE. K=4 remains an explicit typed
        secondary variant with its own path.
      c8_consumers: [C-R-DET, C-R-LLM, C-R-NOPROMPT]
    fairness_invariant: >-
      ONE frozen configuration WITHIN a track, shared by every primary generator
      arm of that track. NOT one identical numeric loss vector across two
      structurally different architectures - the two tracks legitimately differ,
      and forcing one vector across them would be a different claim.
    protocol: P3
    selection_tuple: P3_READY   # equal-weight CASIA-dev + MSU-dev, 15.4
    trial_schedule: frozen_m9_schedule   # 3 G1 + 2 G2 + 30 G5; never shortened
    lock: reports/full/c7/DETECTOR_CONFIG_LOCK.json
    lock_shape: "ONE lock, one sub-config per track under `tracks`"
    lock_verifier: prism_fas.pipeline.adapters.c7.verify_detector_config_lock
    lock_verifier_shared_with: C8   # module level; C8 never writes a laxer one
    trainer: "prism_fas.detector.trainer.run_source_only_flow (canonical)"
    bank_reader: prism_fas.detector.c6_bank.C6MatchedBankReader
    input_verifier: prism_fas.evaluation.c6_evidence.verify_c6_evidence
    binds_into_every_trial_and_the_lock:
      - the search-decision identity
      - the DET bank's selected_set_sha256 and its bank-lock path
      - the C6 selector identity, selected profile and threshold identity
      - the source package identity
      - the track, the typed variant identity and the decision_graph_hash
      - the winning config SHA, its checkpoint SHA and its source_dev calibration
    no_treatment_arm_feedback: >-
      before DETECTOR_CONFIG_LOCK closes, no RND or LLM detector performance and
      no arm comparison is computed or read. The selection is a function only of
      frozen DET training evidence, source-only dev evidence, the frozen
      selection tuple, the canonical tie-break and the frozen envelope.

  # --- what C8 now is -------------------------------------------------------
  c8:
    scientific_executor: IMPLEMENTED
    scientific_status: NOT_RUN
    workflow_dispatch: >-
      _engineering_workflow / _scientific_workflow, chosen on the context
    matrix: >-
      42 rows, from prism_fas.evaluation.source_matrix.build_plan
    consumes: >-
      the C7 lock's per-TRACK sub-config, resolved by row.track
    per_arm_configuration: false
    outputs: [C8_MATRIX_PLAN, C8_SCHEDULE, per-row run_manifest, checkpoint,
              calibration, history, complexity, resources,
              C8_SOURCE_MATRIX_RESULTS, C8_CROSS_SOURCE_DIAGNOSTICS,
              C8_CALIBRATION_STABILITY, C8_TARGET_ISOLATION, C8_ACCEPTANCE]
    selection: >-
      source_dev of the row's own protocol only
    cross_source: >-
      diagnostic only, at the row's own frozen temperature and threshold
    p3_cross_source: >-
      none. A P3-ready row's test domain is the held-out target, predicted at C11

  # --- what C9 now is -------------------------------------------------------
  c9:
    scientific_executor: IMPLEMENTED
    scientific_status: NOT_RUN
    evidence_loader: prism_fas.evaluation.source_evidence.load_row_evidence
    reads: >-
      real C8 run manifests, re-hashing the checkpoints they name
    constructed_evidence_under_science: REFUSED   # _complete_evidence raises
    still_blocked_by: DETECTOR_RELIABILITY_LOCK_C
    blocked_because: >-
      DETECTOR_BA_SEP_PROBE_PROTOCOL, DETECTOR_BA_SEP_EVIDENCE_VECTOR and
      DETECTOR_BA_SEP_PROBE_SEEDS remain NEEDS_SCIENTIFIC_DECISION. They may not
      be chosen from C8 outcomes. C8 may finish; C9 stays blocked.

  c10_c11_c12:
    scientific_executor: NOT_IMPLEMENTED
    fixture_leak_guards: IMPLEMENTED
    note: >-
      each stage's fixture producer now refuses under a scientific context. No
      target byte was opened by this milestone.

  # --- defects this milestone found, before any GPU hour --------------------
  rehearsal_discovered_defects:
    classification: ENGINEERING_FAILURE / PRE-SCIENTIFIC_REHEARSAL
    not_scientific_negative_results: true
    index: reports/evidence/NEGATIVE_EVIDENCE_INDEX.json
    found:
      - id: C8_FIXTURE_EXECUTOR_REACHABLE_UNDER_SCIENCE
        severity: highest
        detail: >-
          C8 had ONE workflow whose row executor called audit_batch and
          build_audit_detector unconditionally. The scheduler was already correct
          - under a scientific context ExecutionContext.limit returns the full 42
          and never reads SMOKE_ROWS - so --profile full would have trained all
          42 rows on fixture batches through an audit model and written 42 PASS
          manifests for C9 to freeze. Every check would have passed.
      - id: C7_SCIENTIFIC_LOCK_ASSERTED_ABSENT
        detail: >-
          C7's only search path used a deterministic analytic objective and
          asserted that reports/full/c7/DETECTOR_CONFIG_LOCK.json does NOT exist,
          while C8 declares that exact path as a required input. The full profile
          could never legitimately produce the lock C8 needs.
      - id: C10_FIXTURE_LABELS_WRITABLE_INTO_SEALED_TARGET
        detail: >-
          _build_fixture mkdir-ed every root sources.target_roots returned and
          wrote a labels.json of invented labels into it. Under a scientific
          context those are the REAL sealed target roots.
      - id: SEARCH_ENGINE_COMPLETED_PASS_RESUME
        detail: >-
          resuming a COMPLETED coordinate pass re-walked the coordinates with
          best restored to the final winning vector, so the EARLY coordinates'
          trials were regenerated with the LATE coordinates already moved -
          different configs, different hashes, missing the reuse table. A rerun
          of a finished search would have silently retrained GPU trials, and the
          trials it produced would not be the ones the pass selected from.
        fix: >-
          a COMPLETED state under a matching plan identity is RETURNED, not
          re-walked. L.11 at pass granularity.
      - id: C7_COORDINATE_ORDER_VS_LR_MULTIPLIER
        detail: >-
          the order check compared against the literal learning_rate while the
          approved decision replaces that coordinate in place with
          learning_rate_multiplier. It failed a correctly executing search.
      - id: C7_DUPLICATE_CHECK_KWARG
        detail: >-
          TypeError from two values for decision_logit_name in one check() call.
      - id: C7_LOCK_ANALYTIC_SUBSTRING_CHECK
        detail: >-
          the verifier grepped metrics_source for "analytic", which the honest
          description "no analytic objective" trips over - failing a real lock
          and passing one that omitted the phrase. Now a declared boolean.
      - id: C7_TRIAL_STATUS_VOCABULARY
        detail: >-
          the trial runner returned FAILED, outside coordinate.TRIAL_STATUS, so
          the summary on disk and the leaderboard disagreed about the same
          configuration.
      - id: C8_ROW_HISTORY_CONDITIONAL
        detail: >-
          the per-row history was written only inside a loop over the stage
          lineage, so a row could finish with no history file at all.
      - id: C8_MANIFEST_READ_THE_TRAINERS_VARIANT
        detail: >-
          the run manifest recorded trainer.variant rather than the variant the
          ROW resolved from its preregistered flags. The same object in
          production, but a trainer that substituted a variant would have agreed
          with itself instead of being caught.

  new_modules:
    - src/prism_fas/evaluation/c6_evidence.py       # strict C6 closure verifier
    - src/prism_fas/evaluation/source_selection.py  # 15.4 video-level tuples
    - src/prism_fas/evaluation/source_evidence.py   # real C8 evidence for C9
    - src/prism_fas/detector/c6_bank.py             # C6 matched bank reader
    - src/prism_fas/search/c7_decision.py           # the frozen arm decision
    - src/prism_fas/reporting/negative_evidence.py  # classified failure index
  new_capabilities:
    - >-
      protocol-scoped source domains (19: P1=CASIA, P2=MSU, P3=both) through
      BatchContract.domains, M9TrainingDataset and M9ValidationDataset. The
      both-domains default is carried out of the payload, so every inherited
      Version-B contract and config identity is unchanged.
    - >-
      run_source_only_flow lifted out of modal_m9.py into detector.trainer, so
      C7 trials, C8 rows and the Modal entrypoint share ONE G1-G2-G5-G6 driver.
  audit_artifacts:
    - reports/readiness/C7_C13_PRODUCTION_READINESS.md
    - reports/readiness/C7_C13_PRODUCTION_READINESS.json
    - reports/evidence/NEGATIVE_EVIDENCE_INDEX.json
  audit_generator: scripts/audit_c7_c13_readiness.py   # inspects source, not belief
  audit_classification: ENGINEERING_AUDIT_NOT_SCIENTIFIC_EVIDENCE

  # --- IMPLEMENTATION CORRECTION, before the first C7 scientific trial -------
  c7_track_g_learning_rate_correction:
    defect_id: C7_TRACK_G_LEARNING_RATE_COORDINATE_NOT_SEARCHED
    classification: IMPLEMENTATION_CORRECTION
    sub_classification: PRE_SCIENTIFIC_EXECUTION_DEFECT
    found_on: 2026-08-24
    found_while: >-
      deriving the declared trial count for the C7 GPU handoff. The handoff is
      required to print that count before launch, which is what surfaced it.
    scientific_result_observed: false
    c7_scientific_trials_executed_before_correction: 0
    target_accessed: false
    record: reports/handoff/LR_ANCHOR_DECISION_CORRECTION.json

    root_cause: >-
      search/lr_decision.py derived "is the learning-rate coordinate searched"
      from "does the multiplier expand over more than one parameter group", so
      UNIQUE_INHERITED_ANCHOR produced candidates = () and an INAPPLICABLE
      coordinate, reasoned as "there is no ambiguity to search and no multiplier
      to apply". The second clause is correct - one applicable group has no
      inherited ratio to hold - but the first turned "no user decision needed"
      into "no search performed".
    consequence_if_executed: >-
      Track G would have run 12 trials instead of 15 and frozen config_G's
      learning rate at the inherited anchor without evaluating 0.5x or 2x.
      config_G is what C-G-RND, C-G-DET and C-G-LLM all train at in C8, so every
      Track-G number would have rested on an unsearched coordinate, under a lock
      recording a complete-looking one-pass envelope.
    why_it_survived_review: >-
      the UNIQUE_INHERITED_ANCHOR branch existed but had never been routed into a
      search plan - C4 and C7 Track R are both B_common_multiplier, and Track G's
      scientific search was first wired at 390fcb2. Three C7 plan checks passed
      over it: the coordinate was PRESENT in the frozen order (just
      inapplicable), the executability gate blocks only on AMBIGUOUS skip
      reasons, and the active-terms check inspects loss weights only. A
      pre-existing test, test_track_g_carries_no_multiplier_because_it_needs_none,
      asserted the defect as intended behaviour, and the frozen 2026-08-17 LR
      record captured it in trial_count_change.C7_TRACK_G.note.

    corrected_semantics:
      unique_inherited_anchor: >-
        describes HOW the anchor was resolved - exactly one inherited LR scalar
        is applicable, so no user decision is required to choose one. It says
        nothing about whether the coordinate is searched.
      learning_rate_coordinate: >-
        15.2.2 puts it first in the frozen order with candidates
        anchor x {0.5, 1.0, 2.0}, under EVERY approved interpretation that has an
        applicable inherited anchor.
      representation: >-
        ONE coordinate, learning_rate_multiplier, for both interpretations. No
        second scalar-valued LR coordinate, no independent per-group LR search.
      track_g_effective_learning_rates: {0.5: 5.0e-5, 1.0: 1.0e-4, 2.0: 2.0e-4}
      track_g_anchor_trial_reproduces_inherited: true
      track_g_parameter_groups: [heads]     # backbone_lr controls zero parameters

    identity_change:
      lr_decision_identity_before: 7ef3492263507d4399828089bbe1af79438bc892e50c8ad732585c1d40c8397c
      lr_decision_identity_after: 16800cb4da6167d66ab34f1b444e794ff7ac6b96c3873fb8fcd9eb2a75207e58
      why_it_moved: >-
        LRDecisionRecord.identity_material hashes the canonical SEMANTIC payload
        of every component, and Track G's now truthfully carries its multipliers,
        its coordinate name and its per-multiplier learning rates. Pretending the
        identity was unchanged would mean two different envelopes sharing one.
      decision_config_bytes_changed: false
      why_no_byte_changed: >-
        configs/search/lr_anchor_decision.yaml was already correct. Its values
        (multipliers [0.5, 1.0, 2.0], Track-G anchor head_lr 1.0e-4, Track-G
        parameter_groups [heads]) were right, and its prose says Track G "needed
        no approval" - never that Track G has nothing to search. The approved
        scientific decision did not require correction; only its implementation
        did.
      c4_gpat_search_plan_identity_changed: false
      c4_identity_guard: >-
        71bfff29bfe1e7ba71d083831a0337a6ae6e0dcfc7f7a75eb9e6f3f3a4ac2b6a, held
        byte-stable and asserted by a test. C4 has already executed
        scientifically and C5 renders through the checkpoint its lock names. An
        intermediate version of this correction decorated Coordinate.anchor_source,
        which enters the plan identity, and moved C4's hash for a cosmetic
        reason; both strings are now identical on the multiplier branch.
      c7_track_g_plan_identity_before: 9ce24e12627f198c5378c48e33e8c09ba8f3a0ef39da38bcd7d8c63bd37cf9d1
      c7_track_g_plan_identity_changed: true   # required: 12 -> 15 trials
      c7_track_r_envelope_changed: false       # 24 trials, 1:10 ratio, unchanged
      c7_search_decision_identity_changed: false   # DET, ed4f6b77..., untouched
      supersession_safe_because: >-
        the correction landed before the first scientific C7 trial. No
        measurement was taken under the superseded identity, nothing needs
        regenerating, and no result changes.

    corrected_declaration:
      track_g_declared_trials: 15
      track_r_declared_trials: 24
      total_declared_trials: 39
      epochs_per_trial: 35
      steps_per_epoch: 45
      total_declared_optimizer_steps: 61425
      derived_from: >-
        prism_fas.search.plan.detector_search_plan over the canonical configs,
        asserted in tests rather than hard-coded as the authority
    regressions:
      - tests/pipeline/test_lr_track_g_coordinate.py   # 25, incl. an injection test
      - tests/pipeline/test_portable_runner.py         # the test that asserted it

  # --- FIRST REAL GPU C7 ATTEMPT — engineering failure, not science ----------
  #
  # Recorded in full because the shape of it matters more than the fix: a missing
  # file produced a scientific-looking verdict about detector configurations, and
  # every layer that should have caught it passed.
  c7_first_gpu_attempt:
    attempted_on: 2026-08-24
    code_head: 3f18628511be7d022924b379db0599f5e4f8d87e
    outcome:
      VERIFY_C6_EVIDENCE: PASS 5/5
      C7_SCIENTIFIC_PLAN: PASS 19/19
      SCIENTIFIC_SOURCE_SEARCH: BLOCKED
    track_g_state:
      status: COMPLETED
      search_plan_identity: 6177df34b4fbda945ed2780b0fd4da26d472dca10762ba4b42976709de3feca8
      completed_coordinates: [learning_rate_multiplier, weight_decay, warmup,
                              lambda_syn, lambda_risk]
      results: 15
      status_fail: 15
      finite_valid: 0
      one_shared_cause: >-
        TextCacheError: the frozen recipe text cache is missing under weights;
        looked for recipe_text_cache.npz, m9/recipe_text_cache.npz,
        pretrained/m9/recipe_text_cache.npz
    track_r_state: none          # no Track-R search occurred
    detector_config_lock: none

    classification:
      stage: C7
      category: ENGINEERING_GLOBAL_INPUT_FAILURE
      defect: FROZEN_RECIPE_TEXT_CACHE_MISSING_FROM_PREFLIGHT
      scientific_negative_result: false
      scientific_envelope_exhausted: false
      scientific_candidate_quality_observed: false
      candidates_legitimately_consumed: 0
      target_access: 0
      why: >-
        all 15 configurations failed on one invariant host-level artifact before
        any configuration-specific detector performance could be measured. A
        missing text cache is independent of the learning rate, the weight decay,
        the warm-up and every loss weight, so it may not consume 15 scientific
        search outcomes.

    training_progress_reached:
      claim: >-
        the 15 rows show no completed training stage, no optimizer-step evidence,
        no checkpoint and no source-selection metric.
      how_it_is_established: >-
        NOT assumed from the failure message. scripts/recover_c7_invalid_search_state.py
        reads every C7_TRIAL_SUMMARY.json on the host and reports
        checkpoints_present, rows_with_selection_metrics, completed_training_stages
        and failure_phases. If any of those is non-empty the state is INELIGIBLE
        for recovery and the exact boundary is reported instead. The claim above is
        the expectation; the script is the authority, and it runs on the GPU host
        where the artifacts are.

    four_defects_found:
      - id: C7_FROZEN_RECIPE_TEXT_CACHE_MISSING
        detail: >-
          the cache was pinned in configs/models/m9_detector.yaml all along
          (cache_file_sha256, cache_identity_sha256) but was absent from the
          portable asset inventory, from sources.verify_detector_inputs and from
          C7's semantic preconditions. The bundle validated, weights/ validated,
          SigLIP2 and ConvNeXt validated, and nothing ever asked for it.
      - id: C7_GLOBAL_FAILURE_CONSUMED_SCIENTIFIC_CANDIDATES
        detail: >-
          _run_scientific_trial wrapped everything in one broad except Exception
          and returned a FAIL TrialResult; coordinate_search did the same for
          anything an evaluator raised. Correct for a configuration-specific
          failure, wrong for a global one.
      - id: C7_ENVELOPE_EXHAUSTED_MISCLASSIFIED
        detail: >-
          EnvelopeExhausted subclasses SearchError and the handler order was
          `except SearchError` first, so the specific branch was dead code. A
          genuine envelope exhaustion was reported as
          SEARCH_STATE_IDENTITY_MISMATCH. The two demand opposite operator
          actions.
      - id: C7_LOGICAL_TRIAL_ARTIFACT_COLLISION
        detail: >-
          15 result rows left 11 summary files. Not a lost trial: the anchor
          configuration recurs whenever it wins a coordinate, and the run root was
          keyed by config SHA alone, so the later occurrence overwrote the
          earlier one's provenance.

    fixes:
      preflight: >-
        sources._frozen_recipe_text_cache verifies the cache twice — file SHA-256
        AND re-derived semantic identity through the canonical
        resolve_recipe_text_cache — inside verify_detector_inputs, which C7 and C8
        share. No fallback, no download, no rebuild. The portable asset inventory
        carries it as required_for_gpu_science, so the preflight now names it as a
        missing item before a trial exists.
      failure_taxonomy: >-
        a typed FatalDependencyError, raised by the trial runner for a TYPED
        allowlist of global-input exception classes and propagated untouched by
        the engine. The stage catches it as an engineering BLOCK: state preserved,
        no candidate consumed, envelope not reported exhausted.
        Configuration-specific failures still become retained FAIL and non-finite
        metrics still become DIVERGED. No prose matching anywhere.
      exception_order: >-
        FatalDependencyError -> EnvelopeExhausted -> SearchError, with a
        structural regression asserting the order.
      evidence_model: >-
        C7_TRIAL_SUMMARY keyed by config SHA holds the trained configuration;
        C7_TRIAL_OCCURRENCE keyed by track/coordinate/trial index holds the search
        position and references it, declaring config_evidence_reused explicitly.

    corrected_cost_accounting:
      note: >-
        the earlier handoff quoted 39 trials and 61,425 optimizer steps. 39 is the
        LOGICAL occurrence count and was right; 61,425 counted every occurrence as
        a separate training, which it is not.
      track_g: {logical_occurrences: 15, unique_configurations: 11}
      track_r: {logical_occurrences: 24, unique_configurations: 17}
      total: {logical_occurrences: 39, unique_configurations: 28}
      optimizer_steps_unique_trainings: 44100
      optimizer_steps_if_every_occurrence_retrained: 61425
      derived_from: >-
        prism_fas.pipeline.adapters.c7._unique_configurations, which replays the
        engine's own coordinate walk over the canonical plan. Not hard-coded.

    recovery:
      script: scripts/recover_c7_invalid_search_state.py
      wired_into_train_py: false     # a person runs it, deliberately, after reading
      preserves: [search state bytes + sha256, every trial summary + sha256,
                  gpu_c7_*.log if present]
      preserved_under: reports/evidence/quarantine/
      clears: only reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_<TRACK>.json
      refuses_if: >-
        any finite-valid row, any PASS row, rows with differing or unrecognised
        causes, a mismatched search-plan identity, an existing
        DETECTOR_CONFIG_LOCK, or artifacts showing real training progress.
      not_touched: [C7 search arm DET, the §15.2.2 envelope, the LR decision,
                    C6, C5]
      not_a_result_driven_restart: >-
        there was no configuration-specific scientific result to restart from.

    frozen_artifact_status:
      artifact: recipe_text_cache.npz
      file_sha256: bb7d3fb4b82ad6ac89ebb06eeac9eb679e2fbb3bab500112cd1e304c187683aa
      cache_identity_sha256: 10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141
      present_on_this_laptop: false
      searched: >-
        the whole working tree; no file named recipe_text_cache* exists anywhere
        outside .git.
      may_be_rebuilt: false
      why_not: >-
        encoding the 128 frozen descriptions is deterministic within one
        environment but not bit-identical across torch/transformers builds or
        across CPU and GPU, so a rebuild yields a different content identity for
        the same science.
      recovery_source: >-
        the historical M9 artifact store, canonically
        /vol/models/pretrained/m9/recipe_text_cache.npz
      consequence: C7 GPU RERUN = BLOCKED until the exact bytes are restored

  # --- the regression that authorized this commit ---------------------------
  tests:
    measured_at_utc: 2026-08-24
    broad_exact_command: >-
      python -m pytest -q --no-header -p no:cacheprovider
      --continue-on-collection-errors
    broad: {passed: 2985, failed: 7, skipped: 104, seconds: 2264}
    c7_and_pipeline_exact_command: >-
      python -m pytest tests/c7 tests/pipeline -q -p no:cacheprovider
    c7_and_pipeline: {passed: 1548, failed: 0, skipped: 3, seconds: 1743}
    inherited_known_failures: 7
    inherited_failure_set_identical: true   # compared test-id by test-id
    new_unexplained_failures: 0
    comparison_source: reports/c0/C0_TEST_SUITE.json
    observed_failures:
      - tests/test_m2_validation.py::test_actual_small_acceptance_validation_passes
      - tests/test_m2_validation.py::test_status_reports_expected_counts
      - tests/test_m8_gpat_synthetic_bank.py::test_pair_plan_lock_records_identities_and_seed
      - tests/test_m8_gpat_synthetic_bank.py::test_pair_plan_identity_excludes_non_portable_fields
      - tests/test_m10_closure.py::test_synthetic_exposure_is_derived_from_the_audited_batch_contract
      - tests/test_m10_closure.py::test_backend_parity_is_reported_as_measured_not_as_a_pass
      - tests/test_m10_target_evaluation.py::test_isolation_declarations_do_not_false_positive
    all_seven_are_missing_version_b_build_products: true
    new_tests_this_milestone: 188
    new_test_files:
      - tests/pipeline/test_c6_evidence_and_bank.py        # 26
      - tests/pipeline/test_c7_scientific_path.py          # 27
      - tests/pipeline/test_c7_search_arm_decision.py      # 30
      - tests/pipeline/test_c8_scientific_path.py          # 16
      - tests/pipeline/test_c9_scientific_evidence.py       # 16
      - tests/pipeline/test_lr_track_g_coordinate.py        # 25
      - tests/pipeline/test_c7_global_input_preflight.py    # 32
      - tests/pipeline/c6_bank_fixture.py                  # shared, not collected
    extended_test_files:
      - tests/pipeline/test_scientific_fixture_leakage.py  # +13 anti-leak regressions
      - tests/pipeline/test_search_engine.py               # +3 resume regressions
    offline: >-
      unchanged. Sockets blocked and ambient credentials deleted by an autouse
      fixture; no provider, no Modal, no GPU and no target label is reachable from
      any test added by this milestone.

  unchanged_by_this_milestone:
    version_b: >-
      7799f7decd35db6987ce4578824e5bd8d9eab4ae, tag
      m10-blind-evaluation-checkpoint, clean, untouched
    c5: >-
      scientific candidate pool, render logic and lock unchanged
    c6: >-
      CLOSED. No C6 scientific artifact, selector, threshold, profile or bank was
      rewritten. git diff over src/prism_fas/synthesis/, adapters/c5.py,
      adapters/c6.py, reports/ and assets/ is empty, so the selector identity
      27c20b71ff7c1d42dca1f7034f81bfd61400b5a17a2421fe823603c102e17ef3 and the
      three selected_set_sha256 values are unchanged by construction. They could
      not be re-derived here: reports/full/c6 lives on the GPU host and is absent
      from this laptop.
    target_firewall: >-
      unchanged; target_access 0 throughout


next_authorized_action: >
  HISTORICAL — SUPERSEDED 2026-08-25. Preserved verbatim below because it is
  what was actually authorized and true when written (2026-08-22): the C5
  scientific render executor did not yet exist. It has since been built, and
  C4, C5, C6 and C7 have all executed scientifically and closed on the GPU
  host. For the CURRENT next authorized action see
  `c7_scientific_closure_reconciliation.next_authorized_action` below — in
  summary: C8 is READY_NOT_RUN, pending explicit user authorization, at the
  exact command in `reports/readiness/C8_GPU_SCIENTIFIC_HANDOFF.md`.

  BUILD THE C5 SCIENTIFIC RENDER EXECUTOR ON TOP OF THE FROZEN PLAN.

  Both C5 scientific decisions are now closed and implemented:
  C5_SOURCE_PAIR_PLAN_V1 exists in prism_fas.synthesis.c5_source_pair_plan, and
  C5 raw generation identity provably binds no C6 calibration. The schedule is
  arm-independent by construction and every frozen cardinality is asserted on the
  plan itself rather than promised.

  What remains, in dependency order:

      1. the mechanical generation/evaluation separation of the canonical
         SyntheticBankGenerator, preserving the Version-B class, API and
         identities byte-for-byte
      2. the scientific C5 adapter branch and its substages — VERIFY_C4_LOCK,
         FREEZE_SOURCE_PAIR_PLAN, BUILD_ARM_CANDIDATE_PLANS, RENDER_PHYSICS,
         RENDER_GPAT, VERIFY_RAW_CANDIDATES, FINALIZE_C5
      3. the C4 GPAT_CONFIG_LOCK verification path, reusing C4's own strict
         semantics rather than a second implementation
      4. per-candidate atomic terminal records, resume by generation identity and
         output hash, and failure retention that never resamples
      5. the C5 scientific completion lock over all 6144 positions

  Nothing in 1-5 requires a new scientific decision; the frozen plan supplies
  every input they need. They do require a GPU to exercise, which this laptop
  does not have.

  MEANWHILE C4 IS STILL UNBLOCKED. The GPU host can run
  `/usr/bin/python3 train.py` at this HEAD to execute the scientific C4 search.
  C5 will block afterwards, which remains correct.

  What review is being asked to confirm: the frozen plan is exactly the schedule
  the decision specified — p = 8r+s, LIVE_LIST[p mod N], even slots Physics and
  odd slots GPAT, GPAT slots 1/5 same-domain and 3/7 cross-domain, spoof selected
  by the position-keyed digest over the sorted eligible pool — and that the
  schedule takes no arm, no recipe bank and no recipe content as input.

  What review must NOT read into it: no candidate was rendered, no bank was
  generated, no C6 threshold was chosen or fitted, and the Version-B
  candidate_plan and SyntheticBankGenerator are byte-identical to before.

# ==========================================================================
# C7 SCIENTIFIC CLOSURE — PROJECT_STATE RECONCILIATION (this milestone)
# ==========================================================================
#
# DOCUMENTATION-ONLY. No C7 or C8 execution, no target access, and no change
# to any scientific source/config/search logic happened while writing this
# section. It reconciles the global fields above (current_milestone,
# scientific_status, execution_pipeline, next_authorized_action) with the
# scientific facts already recorded in c7_c13_readiness_milestone above and
# in the two closure/handoff artifacts this section points at. It does not
# recompute or restate any identity that was not already recorded elsewhere;
# every SHA-256 below is copied, not derived.
c7_scientific_closure_reconciliation:
  milestone: C7_SCIENTIFIC_CLOSURE_AND_REPORTING_RECONCILIATION
  reconciled_on: 2026-08-25
  execution_profile: none   # documentation reconciliation only
  scientific_execution: NONE
  target_access: 0
  gpu_used: false
  source_code_head: 784e517d6035ece338718a384c170d8ab774c8f4
  reconciles_against:
    - reports/readiness/C7_SCIENTIFIC_CLOSURE_AUDIT.md
    - reports/readiness/C8_GPU_SCIENTIFIC_HANDOFF.md
    - reports/readiness/C7_GPU_SCIENTIFIC_HANDOFF.md   # pre-launch handoff; superseded for status, kept as history

  # --- per-stage table -------------------------------------------------------
  stage_table:
    c3:
      status: SCIENTIFICALLY_COMPLETE_FROZEN
      note: live generation, frozen bank at branch_point 36f10fd24880a0bfb3e6c3c2ba8a3fcc53195572
    c4:
      status: SCIENTIFICALLY_COMPLETE_FROZEN
      artifact: reports/full/c4/GPAT_CONFIG_LOCK.json
      note: see first_scientific_gpu_run.c4 above (scientific_status PASS)
    c5:
      status: SCIENTIFICALLY_COMPLETE
      note: >-
        real synthesis pool generated on the GPU host: 6082/6144 generated,
        62 semantic_failed, 0 runtime_unresolved (first_scientific_gpu_run.c5
        above). Terminal and complete under the corrected C5 acceptance rule.
    c6:
      status: SCIENTIFIC_CLOSED_FROZEN
      matched_banks: [RND, DET, LLM]
      selected_profile: NOMINAL
      selector_identity_sha256: 27c20b71ff7c1d42dca1f7034f81bfd61400b5a17a2421fe823603c102e17ef3
      quality_threshold_identity_sha256: 8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19
      det_bank_selected_set_sha256: 32f6e0e129da277d0be76abc0758bc3d63ad16caba111efc801ac92de291f5b0
      target_access: 0
    c7:
      status: SCIENTIFIC_CLOSED_VALID
      retraining_required: false
      target_access: 0
      search_decision:
        training_arm: DET
        per_arm_search: false
        search_decision_identity: ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e
      lr_decision_identity: 16800cb4da6167d66ab34f1b444e794ff7ac6b96c3873fb8fcd9eb2a75207e58
      source_package_identity: 08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9
      track_g:
        logical_trials: 15
        pass: 15
        winner_config_sha256: 97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd
        winner_checkpoint_sha256: 08e4f10083990bfbb846761d8d63ac82e5bdadb2765ae194ce6ae992c680133e
        winner_epoch: 6
        lr_interpretation: UNIQUE_INHERITED_ANCHOR
        learning_rate_multiplier: 2.0
        head_lr_anchor: 1.0e-4
      track_r:
        logical_trials: 24
        pass: 24
        winner_config_sha256: d8537b257b84d394f0a43085bba2dc2fb1cafb2533ac199d172001df6d631110
        winner_checkpoint_sha256: 8c81c1d16b066ab3a2e88d0dae0c43d182a09b751b9c6cb699a75f67c0d8c2be
        winner_epoch: 11
        lr_interpretation: B_common_multiplier
        backbone_lr_anchor: 1.0e-5
        head_lr_anchor: 1.0e-4
        preserved_ratio: [1.0, 10.0]
        learning_rate_multiplier: 2.0
      reporting_closure:
        inherited_anchor_report_semantics: PRE_DECISION_STRUCTURAL
        is_the_final_lr_execution_state: false
        final_lr_execution_state_fields: [lr_interpretation, lr_anchor_vector,
                                          lr_decision_resolution, lr_decision_identity]
        fix_landed_at: 784e517d6035ece338718a384c170d8ab774c8f4
        scientific_result_remains_valid: true
      first_attempt:
        code_head: 3f18628511be7d022924b379db0599f5e4f8d87e
        classification: ENGINEERING_GLOBAL_INPUT_FAILURE
        defect: FROZEN_RECIPE_TEXT_CACHE_MISSING_FROM_PREFLIGHT
        scientific_negative_result: false
        candidates_consumed: 0
        target_access: 0
        quarantined_under: reports/evidence/quarantine/
        distinct_from_the_completed_run_above: true
    c8:
      status: SCIENTIFIC_CLOSED_VALID
      scientifically_run: true
      outcome: PASS
      substages:
        VERIFY_INPUTS: {status: PASS, checks: "29/29"}
        PLAN_MATRIX: {status: PASS, checks: "5/5"}
        SCHEDULE: {status: PASS, checks: "4/4"}
        EXECUTE_ROWS: {status: PASS, checks: "6/6"}
        CROSS_SOURCE_DIAGNOSTICS: {status: PASS, checks: "2/2"}
        CALIBRATION_STABILITY: {status: PASS, checks: "2/2"}
        TARGET_ISOLATION: {status: PASS, checks: "3/3"}
        ACCEPTANCE: {status: PASS, checks: "8/8"}
      rows: {total: 42, pass: 42, fail: 0}
      protocol_distribution: {P1: 9, P2: 9, P3: 24}
      matrix:
        P1: {C-G-RND: 3, C-G-DET: 3, C-G-LLM: 3}
        P2: {C-G-RND: 3, C-G-DET: 3, C-G-LLM: 3}
        P3: {C-G-RND: 5, C-G-DET: 5, C-G-LLM: 5, C-R-DET: 3, C-R-LLM: 3, C-R-NOPROMPT: 3}
      target_access: 0
      gpu_artifacts_live_only_on_gpu_host: [runs/full/c8/, reports/full/c8/,
                                            reports/full/c8/C8_FULL.json,
                                            reports/full/c8/C8_ACCEPTANCE.json]
      not_invented_on_this_laptop: true
      closure_audited_in: reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md
      pre_launch_engineering_defects_found_and_fixed:
        - id: EXPLICIT_PREFLIGHT_ONLY_NOT_HONORED
          found_on: 2026-08-25
          found_before_first_c8_scientific_run: true
          detail: >-
            `train.py::_explicit` parsed --preflight-only into args.preflight_only
            and never read it; it called orchestrator.run(...) unconditionally, so
            the handoff's advertised read-only preflight command
            (`python train.py --profile full --from C8 --to C8 --preflight-only`)
            was not guaranteed read-only and could have reached
            C8Adapter.workflow() and trained a real detector.
          fix: >-
            preflight_only threaded onto AdapterRequest, through
            orchestrator.run(), into EngineeringAdapter.run() (C4-C13); stops
            before workflow() after the SAME full_precondition_gate a real run
            applies; orchestrator.run() returns before _write_reports/record/
            write_state when preflight_only, so no reports/full/c8 artifact, no
            MASTER_RUN_INDEX row and no PIPELINE_STATE mutation can occur.
          regression_tests: tests/pipeline/test_explicit_preflight_only.py
          c8_rows_executed_by_the_defect: 0
          detector_trained: false
          target_access: 0
          closure_record: reports/readiness/C8_GPU_SCIENTIFIC_HANDOFF.md
        - id: C8_PRELAUNCH_REQUIRED_INPUT_ROOT_DRIFT
          found_on: 2026-08-25
          found_by: >-
            the first real, corrected, read-only C8 preflight, run on the GPU
            host: `python train.py --profile full --from C8 --to C8 --resume
            --preflight-only` reported FULL_PRECONDITION_GATE BLOCKED,
            checks 6/7, failing c8_input_pretrained_weights
            (data/packages/pretrained is absent) — the preflight itself
            behaved correctly; the path it checked was wrong.
          found_before_first_c8_scientific_run: true
          false_required_paths: [data/packages/pretrained, data/packages]
          canonical_roots: [sources.WEIGHT_ROOT (weights),
                            sources.SOURCE_PACKAGE_ROOT
                            (data/packages/prism_data_v1_m3b)]
          detail: >-
            C8Adapter.required_inputs() declared pretrained_weights at
            data/packages/pretrained (nothing ever writes this path) and
            source_packages at data/packages (the M3B package's PARENT, not
            the package). C7Adapter.required_inputs() already used the
            correct roots and C7's completed GPU scientific run actually
            trained against them. Deeper gap: C8's gate only ever checked
            presence, never the SHA-verified pinned weights or the frozen
            recipe text cache C7's own _scientific_prepare already required
            via sources.verify_detector_inputs.
          fix: >-
            required_inputs() for both C7 and C8 now import
            sources.WEIGHT_ROOT, sources.SOURCE_PACKAGE_ROOT,
            sources.C5_CANDIDATES_ROOT, c6_evidence.C6_REPORTS and
            c7.SCIENTIFIC_CONFIG_LOCK_PATH instead of hand-spelling them, so
            the two stages cannot independently drift. C8Adapter
            .semantic_preconditions() now also calls
            sources.verify_detector_inputs() - the same canonical,
            SHA-verifying check C7 already used - so the gate genuinely
            validates the pinned weights, the text cache and the package, not
            merely their presence. data/packages/pretrained was NOT created;
            hiding the drift was explicitly rejected as a fix.
          regression_tests: tests/pipeline/test_c8_precondition_root_drift.py
          c8_rows_executed_by_the_defect: 0
          detector_trained: false
          target_access: 0
          closure_record: reports/readiness/C8_GPU_SCIENTIFIC_HANDOFF.md
    c9:
      status: BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE
      blocker: DETECTOR_RELIABILITY_LOCK_C
      lock_exists: false   # not yet REGISTERED anywhere (laptop has no real artifacts;
                           # GPU registration has not been run either) — see
                           # ba_sep_option1_v2_scientific_failure_closure below
      decision_audit: reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md
      decision_audit_json: reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.json
      audit_verdict: >-
        the barrier is a correctly-designed, correctly-enforced fail-closed
        gate, not a defect. Confirmed spec-vs-implementation incompatibility:
        the historical Version-B evidence vector [p_global, s_region, nine
        regional distances] is not obtainable from any of the 42 existing C8
        rows (Track G is global-only by NORMATIVE contract; primary Track R
        trains with manifold OFF; no C-R-RND row exists).
      ba_sep_option1_protocol_freeze:
        status: SUPERSEDED_BY_V2_PREEXECUTION_CORRECTION
        decision_id: C9_DETECTOR_BA_SEP_OPTION1_V1
        approved_by: explicit user approval, before any BA_sep value was observed
        resolves_test: synthetic_vs_real_spoof_probe   # one of nine required tests
        config: configs/evaluation/c9_detector_ba_sep_option1_v1.yaml   # untouched, historical
        protocol_identity: a6da0ce75ebd92589ea61cba24a85bf8d8144bdbb99f7ec54d31066a66594908
        defect_found_before_execution: >-
          declared matched_source_split.group_safe: true while actually
          partitioning on sample_id/synthetic_id (SAMPLE identity) rather
          than source_record_id (GROUP identity); see
          ba_sep_option1_v2_preexecution_correction below
        real_ba_sep_computed: false
        record: reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.md
        record_json: reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.json
        tests: tests/pipeline/test_c9_ba_sep_option1_protocol.py   # 44 passed, updated for the V2 PopulationRecord shape
      ba_sep_option1_v2_preexecution_correction:
        status: FROZEN_NOT_RUN
        decision_id: C9_DETECTOR_BA_SEP_OPTION1_V2
        supersedes: C9_DETECTOR_BA_SEP_OPTION1_V1
        supersession_reason: PRE_EXECUTION_GROUP_IDENTITY_CORRECTION
        approved_by: explicit user approval, before any BA_sep value was observed (under V1 or V2)
        resolves_test: synthetic_vs_real_spoof_probe   # one of nine required tests
        config: configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
        protocol_identity: 720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
        superseded_protocol_identity: a6da0ce75ebd92589ea61cba24a85bf8d8144bdbb99f7ec54d31066a66594908
        implementation: src/prism_fas/evaluation/synthetic_real_probe.py
        runner: src/prism_fas/evaluation/synthetic_real_probe_runner.py   # see ba_sep_option1_v2_runner_integration_fix
                                                                          # below: the --arm-per-mode CLI this freeze
                                                                          # first shipped was replaced there
        evidence_vector: [global_logit_G, p_global]   # unchanged from V1
        checkpoints_bound: 15   # 5 seeds x {RND,DET,LLM} P3-ready Track-G rows; identities
                                # empty here, bound only on a host with runs/full/c8/
        probe_seed_values: [20260806, 20260807, 20260808]   # unchanged from V1
        ba_ceiling: 0.75   # unchanged from V1
        hard_verdict_rule: BA_sep_RND<=0.75 AND BA_sep_DET<=0.75 AND BA_sep_LLM<=0.75
        group_identity_correction: >-
          PopulationRecord now carries sample_identity (selection axis) and
          stable_group_identity (split axis, always source_record_id)
          separately; synthetic rows resolve stable_group_identity via
          live_target_sample_id -> source_train row -> source_record_id,
          fail-closed if unmappable
        real_ba_sep_computed: false
        run_scientific_probe_wired: false   # SUPERSEDED — see ba_sep_option1_v2_runner_integration_fix:
                                            # execute_joint_probe is now real code
        record: reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md
        record_json: reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.json
        tests: tests/pipeline/test_c9_ba_sep_option1_v2_runner.py   # extended; see below for the current count
      ba_sep_option1_v2_runner_integration_fix:
        status: IMPLEMENTED_NOT_RUN
        v2_protocol_identity_unchanged: 720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
        v3_created: false
        defects_fixed:
          - id: PER_ARM_BIND_CONTRADICTS_JOINT_BALANCE
            problem: >-
              --bind-only --arm {RND,DET,LLM} bound one arm at a time, which
              forced N = min(real, this_arm, 0, 0) = 0 under the already-frozen
              JOINT balancing rule; never a scientific decision, and no real
              bind had ever succeeded against real data on any host
            fix: >-
              --bind-only now takes no --arm; build_checkpoint_binding /
              build_population_plan resolve and balance all three arms jointly
              in one call
          - id: EXECUTE_NOT_REAL_CODE
            problem: --execute routed to the deliberately-NotImplementedError single-arm run_scientific_probe
            fix: >-
              execute_joint_probe reuses the exact C8 row-construction path
              (C7 lock, C6 bank, _detector_config_for_row, M9Trainer,
              checkpoint.load_checkpoint/apply_checkpoint) to strict-load all
              15 checkpoints and forward evidence through the unchanged frozen
              mechanics
        cli: python -m prism_fas.evaluation.synthetic_real_probe_runner --repo . {--preflight-only|--bind-only|--execute}
        no_arm_flag_on_any_mode: true
        joint_execution_binding_path: reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_EXECUTION_BINDING.json
        joint_population_plan_path: reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_POPULATION_PLAN.json
        result_artifacts_on_a_real_execute:
          - reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_RESULT.json
          - reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PER_SEED.json
          - reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PROBE_PARAMETERS.json
          - reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_EVIDENCE_MANIFEST.json
          - reports/full/c8/reliability/synthetic_vs_real_spoof_probe/SYNTHETIC_VS_REAL_SPOOF_PROBE_VERDICT.json
        zero_sample_cell_fails_closed: true
        atomic_two_artifact_bind: true
        reuse_refuses_on_mismatched_existing_binding: true
        preflight_strengthened_to_all_inputs_required: true
        execute_distinguishes_pass_fail_blocked_usage_exit_codes: true
        scientific_failed_verdict_is_a_real_result_written_honestly: true
        latent_bugs_also_fixed:
          - resolve_checkpoint_set's hand-built checkpoint path was wrong (fixed to read
            checkpoint.path/kind off the manifest); never affected a real binding
          - checkpoint_binding_identity/population_plan_identity did not exclude their own
            field from their own hash material (self-reference bug); never affected a real artifact
        runner_dry_run_on_this_laptop:
          preflight_only: {exit_code: 2, ready_for_bind: false, target_access: 0}
          bind_only: {exit_code: 2, artifacts_written: false, reason: no M3B package, no runs/full/c8/ on this host}
          execute: {exit_code: 2, executed: false, reason: no execution binding on disk; run --bind-only first}
        real_ba_sep_computed: false
        record: reports/readiness/C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.md
        record_json: reports/readiness/C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.json
        tests:
          v1_shared_mechanics_file: tests/pipeline/test_c9_ba_sep_option1_protocol.py   # 44 passed
          v2_file: tests/pipeline/test_c9_ba_sep_option1_v2_runner.py   # 90 passed
          all_c9_scoped_tests_total: 150
          broad_regression_tests_c7_and_pipeline: {failed: 33, passed: 1663, skipped: 22, baseline_unchanged: true}
      ba_sep_option1_v2_gpu_binding:
        # Reported by the user as already true at the start of the
        # inference-parity-fix task; recorded here as the real, current GPU
        # scientific state. This laptop cannot verify it directly (it has no
        # runs/full/c8/); it is authoritative because it was named as fact
        # by the human directing this work, not derived on this laptop.
        status: FROZEN_BOUND
        host: GPU_SCIENTIFIC_HOST   # not this laptop
        protocol_identity: 720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
        checkpoint_binding_identity: fa380fa8e732f8536fe175d449542e636563d92d8d75f64bb07b40ca180f63b0
        population_plan_identity: 90d00d9f4bb50a93724d1ac6a632d6fa5052cf2d7ec0d08989c4c7004fa6cae1
        checkpoints_bound: 15
        population_cells_bound: 12
        real_ba_sep_computed: false
        real_checkpoint_loaded_for_the_probe: false   # binding hashes manifests; it never opens weights
        expected_to_remain_reproducible_after_the_inference_parity_fix: true   # build_checkpoint_binding
                                                                               # and build_population_plan
                                                                               # were not edited
      ba_sep_option1_v2_inference_parity_fix:
        status: IMPLEMENTED_NOT_RUN
        v2_protocol_identity_unchanged: 720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
        v3_created: false
        discovered: after the real GPU binding above succeeded, before the first real --execute
        gaps_fixed:
          - id: HARD_CODED_CPU_DEVICE
            problem: construct_row_trainer passed device="cpu" unconditionally instead of C8's own scientific device resolver
            fix: construct_row_trainer now calls pipeline.adapters.c7._scientific_device() first, before the C7 lock or C6 bank are touched, and passes the resolved device into M9Trainer; fails closed if no CUDA
          - id: CUDA_UNSAFE_EVIDENCE_CONVERSION
            problem: extract_evidence called np.asarray(tensor) directly, which raises on a CUDA tensor
            fix: new _evidence_scalar() helper does .detach().float().cpu().numpy() first for any torch.Tensor; linear probe itself remains CPU float64, unchanged
          - id: ONE_SAMPLE_PER_FORWARD
            problem: forward_evidence_for_records forwarded one sample per model call, not matching C8's own batched cross-source evaluation
            fix: now chunks by trainer.config.validation_batch_size, forwards under torch.no_grad(), slices per-sample evidence via the unchanged extract_evidence -- no sample reordered or mixed across chunks
        new_explicit_pre_forward_guard: >-
          execute_joint_probe now calls sources.verify_detector_inputs(repo, arms=ARMS)
          and requires the CURRENT source package identity and all three CURRENT
          C6 arm bank identities to match checkpoint_binding's bound values,
          BEFORE any checkpoint construction or forward -- previously only ever
          reachable as an eventual per-checkpoint RunIdentity refusal
        build_checkpoint_binding_and_build_population_plan_edited: false
        expected_gpu_rerun_result: >-
          --preflight-only then --bind-only on the GPU host should reproduce
          the SAME three identities recorded in ba_sep_option1_v2_gpu_binding
          above, with reused: true
        real_ba_sep_computed: false
        runner_dry_run_on_this_laptop:
          preflight_only: {exit_code: 2, target_access: 0}
          bind_only: {exit_code: 2, artifacts_written: false, reason: no CUDA, no M3B package, no runs/full/c8/ on this host}
          execute: {exit_code: 2, executed: false, reason: no execution binding on disk; run --bind-only first}
        record: reports/readiness/C9_BA_SEP_OPTION1_V2_INFERENCE_PARITY_FIX.md
        record_json: reports/readiness/C9_BA_SEP_OPTION1_V2_INFERENCE_PARITY_FIX.json
        tests:
          v1_shared_mechanics_file: tests/pipeline/test_c9_ba_sep_option1_protocol.py   # 44 passed
          v2_file: tests/pipeline/test_c9_ba_sep_option1_v2_runner.py   # 113 passed
          all_c9_scoped_tests_total: 173
          broad_regression_tests_c7_and_pipeline: {failed: 33, passed: 1686, skipped: 22, baseline_unchanged: true}
      ba_sep_option1_v2_scientific_failure_closure:
        # THE scientific result. Reported by the user as already observed on
        # the GPU host, in this task; this laptop has not independently read
        # the real result files (OBSERVED_ON_GPU_USER_REPORTED, not
        # GPU_ARTIFACT_REGISTRATION_VERIFIED — that verification is what
        # detector_reliability_runner --register-ba-sep-result performs, and
        # it has not been run on the GPU host yet).
        status: SCIENTIFICALLY_RUN
        execute_exit_code: 1   # scientific execution COMPLETED, hard verdict FAIL
        v2_protocol_identity_unchanged: 720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
        v3_created: false
        ba_sep_by_arm: {RND: 0.7843079833902619, DET: 0.8514170182841069, LLM: 0.7902658339472685}
        ba_ceiling: 0.75
        hard_verdict_rule: "PASS iff BA_sep_RND<=0.75 AND BA_sep_DET<=0.75 AND BA_sep_LLM<=0.75"
        per_arm_verdict: {RND: FAIL, DET: FAIL, LLM: FAIL}
        overall_probe_verdict: FAILED
        per_seed_ba_sep:
          RND: {20260806: 0.753968253968254, 20260807: 0.8164556962025317, 20260808: 0.7825}
          DET: {20260806: 0.8333333333333333, 20260807: 0.8734177215189873, 20260808: 0.8474999999999999}
          LLM: {20260806: 0.7486772486772486, 20260807: 0.819620253164557, 20260808: 0.8025}
        single_seed_below_ceiling_irrelevant_to_aggregate_rule: true   # e.g. RND/LLM's own
                                                                       # 20260806 seed, individually
        rerun_forbidden: true
        threshold_tuning_forbidden: true
        seed_replacement_forbidden: true
        population_replacement_forbidden: true
        checkpoint_cherry_picking_forbidden: true
        c6_bank_reopening_under_current_protocol_forbidden: true
        negative_result_retained: true
        no_rerun_guard:
          implementation: prism_fas.evaluation.synthetic_real_probe_runner._execute
          states: [none_present->normal_first_execution, all_present->reused_no_recompute,
                   partial->BLOCKED_PARTIAL_SCIENTIFIC_RESULT_SET]
          verifier: prism_fas.evaluation.synthetic_real_probe.validate_existing_scientific_result
        c_h4:
          ba_sep_llm: 0.7902658339472685
          llm_le_ceiling: false
          llm_beats_det: true
          llm_beats_rnd: false
          basic_conditions_hold: false
          status: NOT_SUPPORTED_BY_CURRENT_BA_SEP_RESULT
          bootstrap_fabricated: false
        record: reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.md
        record_json: reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.json
        tests:
          new_file: tests/pipeline/test_c9_ba_sep_scientific_failure_closure.py   # 44 passed
          all_c9_scoped_tests_total: 217
          broad_regression_tests_c7_and_pipeline: {failed: 33, passed: 1730, skipped: 22, baseline_unchanged: true}
      ba_sep_option1_v2_failed_barrier_registration:
        status: PENDING_GPU_VERIFICATION   # implemented, never run against real GPU artifacts
        cli: python -m prism_fas.evaluation.detector_reliability_runner --repo . {--status|--register-ba-sep-result}
        binds_only: synthetic_vs_real_spoof_probe
        other_eight_tests_left: UNRESOLVED
        uses_existing_lock_payload: true   # no second lock schema
        idempotent: true
        refuses_to_overwrite_different_existing_lock: true
        never_mutates_the_seven_ba_sep_artifacts: true
        detector_checkpoint_identities_bound_from: C9_BA_SEP_EXECUTION_BINDING.json (row_id -> checkpoint_sha256, all 15)
        expected_barrier_once_registered:
          synthetic_vs_real_spoof_probe: FAILED
          overall: FAILED
          c9_may_close: false
        runner_dry_run_on_this_laptop:
          status: {exit_code: 2, ba_sep_result_available: false, reason: NO_SCIENTIFIC_RESULT_ON_THIS_HOST}
          register: {exit_code: 2, registered: false, reason: BA_SEP_RESULT_ABSENT_OR_INVALID}
        new_structural_validator: detector_reliability.validate_lock_record   # recognizes a valid
                                                                              # FAILED record; is
                                                                              # NOT C9's pass gate
        verify_lock_unweakened: true   # still strictly overall==PASSED-only, confirmed by regression
        c9_rejects_failed_record: true   # proven via C9Adapter.semantic_preconditions directly
      ba_sep_option1_v2_gpu_registration:
        # OBSERVED_ON_GPU_USER_REPORTED — reported to this laptop as already
        # true; this laptop has NOT independently read these GPU files.
        # GPU_ARTIFACT_REGISTRATION_VERIFIED is a DIFFERENT, stronger claim
        # this field does not make.
        status: USER_REPORTED_COMPLETE
        lock_path: reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
        identity_sha256_user_reported: 40825a5fffcbbdd681e5d8b0354e8371dcccabc36b51d1b9a12cd8bd29e73fbe
        registered: true
        reused: false
        overall: FAILED
        c9_may_close: false
        target_access: 0
        validate_lock_record_user_reported: {valid: true, problems: []}
        verify_lock_user_reported: {valid: false, expected_problems:
          ["overall is FAILED, not PASSED", "synthetic_vs_real_spoof_probe is FAILED",
           "remaining required tests unresolved"]}
        laptop_independently_verified: false
      post_ba_failure_decision_dossier:
        status: WRITTEN_NOT_A_REDESIGN
        record: reports/readiness/C9_DETECTOR_RELIABILITY_POST_BA_FAILURE_DECISION_DOSSIER.md
        record_json: reports/readiness/C9_DETECTOR_RELIABILITY_POST_BA_FAILURE_DECISION_DOSSIER.json
        remaining_eight_tests_audit:
          residual_scale_zero: NEEDS_SCIENTIFIC_DECISION
          recipe_region_shift: NEEDS_SCIENTIFIC_DECISION
          artifact_map_swap: NEEDS_SCIENTIFIC_DECISION
          cross_route_synthetic: NEEDS_SCIENTIFIC_DECISION
          benign_jpeg_corruption: NEEDS_SCIENTIFIC_DECISION
          benign_resize_corruption: NEEDS_SCIENTIFIC_DECISION
          benign_color_corruption: NEEDS_SCIENTIFIC_DECISION
          crop_padding_interpolation: STRUCTURALLY_DATA_BLOCKED   # re-confirmed unchanged
        options_compared: [OPTION_A_CLOSE_AS_NEGATIVE, OPTION_B_SOURCE_ONLY_DIAGNOSTICS,
                           OPTION_C_NEW_VERSIONED_REDESIGN_PROTOCOL]
        recommended: OPTION_B_FIRST_THEN_EVALUATE_OPTION_C
        option_b_cannot_reopen_c9: true
        option_c_implemented_by_this_task: false
        no_threshold_invented: true
        no_test_executed: true
      post_failure_source_diagnostics_v1:
        status: PROTOCOL_FROZEN_GPU_EXECUTION_PENDING
        decision_id: C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1
        not_a_ba_sep_revision: true
        not_a_c9_pass_protocol: true
        config: configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
        protocol_identity: cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5
        implementation: src/prism_fas/evaluation/post_failure_diagnostics.py
        runner: src/prism_fas/evaluation/post_failure_diagnostics_runner.py
        cli: python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . {--preflight-only|--bind-only|--status|--execute}
        result_namespace: reports/full/c8/reliability/post_failure_source_diagnostics_v1/   # NEVER
                          # synthetic_vs_real_spoof_probe/ or DETECTOR_RELIABILITY_LOCK_C.json
        re_audited_executability:
          benign_jpeg_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          benign_resize_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          benign_color_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          cross_route_synthetic: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          recipe_region_shift: {classification: NEEDS_SCIENTIFIC_DECISION, gpu_ready: false,
            note: "region_embeddings/region_attention now known available for primary Track R; which existing signal is the heat-map remains undecided"}
          artifact_map_swap: {classification: NEEDS_SCIENTIFIC_DECISION, gpu_ready: false,
            note: "local_logits/attack_region_mask/weighted_local_loss now known available for primary Track R; alignment metric+threshold not frozen"}
          residual_scale_zero: {classification: STRUCTURALLY_MODEL_BLOCKED, gpu_ready: false,
            note: "needs synthesis-time residual control (forbidden) AND region_distances (absent from every primary Version-C checkpoint)"}
          crop_padding_interpolation: {classification: STRUCTURALLY_DATA_BLOCKED, gpu_ready: false,
            note: "re-confirmed unchanged: no bbox, no source-frame path in the frozen package"}
        gpu_ready_count: 4
        benign_corruption_threshold: "calibration_mean_shift + 3 * calibration_std_shift, group-safe 50/50 split on source_record_id"
        cross_route_reuses_ba_sep_ceiling: true   # 0.75, not invented
        checkpoint_policy_identical_to_ba_sep: true   # 5/arm, 15 total, same seeds
        c9_may_close_hard_coded_false_in_every_artifact: true
        runner_dry_run_on_this_laptop:
          preflight_only: {exit_code: 2, target_access: 0}
          bind_only: {exit_code: 2, artifacts_written: false, reason: no M3B package on this host}
          status: {exit_code: 2, diagnostics_result_available: false}
          execute: {exit_code: 2, executed: false, reason: no diagnostics binding on disk}
        real_diagnostic_value_computed: false
        ba_sep_artifacts_untouched_confirmed_by_regression: true
        record: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_PROTOCOL_FREEZE.md
        record_json: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_PROTOCOL_FREEZE.json
        gpu_handoff: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_GPU_HANDOFF.md
        tests:
          new_file: tests/pipeline/test_c9_post_failure_source_diagnostics.py   # 60 passed
          all_c9_scoped_tests_total: 277
          broad_regression_tests_c7_and_pipeline: {failed: 33, passed: 1790, skipped: 22, baseline_unchanged: true}
        status: FROZEN_BUT_SUPERSEDED_BEFORE_FIRST_EXECUTION
        superseded_reason: PREEXECUTION_SCIENTIFIC_AUDIT_FOUND_SEMANTIC_ACCEPTANCE_DEFECTS
        superseded_by: post_failure_source_diagnostics_v2 (below)
        v1_scientific_metric_observed: false
        v1_preserved_byte_for_byte: true   # config, module and runner all git-diff-clean vs HEAD
      post_failure_source_diagnostics_v2:
        # PRE-EXECUTION SCIENTIFIC CORRECTION of V1, found and applied before
        # any GPU --execute ever ran (V1's scientific metric was never
        # observed). V1 is preserved unchanged as historical pre-execution
        # design evidence and is NEVER edited; V2 is a separately versioned,
        # separately namespaced protocol.
        status: PREEXECUTION_CORRECTION
        config: configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml
        protocol_identity: 05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523
        supersedes_v1_identity: cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5
        implementation: src/prism_fas/evaluation/post_failure_diagnostics_v2.py
        runner: src/prism_fas/evaluation/post_failure_diagnostics_v2_runner.py
        cli: python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . {--preflight-only|--bind-only|--status|--execute}
        result_namespace: reports/full/c8/reliability/post_failure_source_diagnostics_v2/   # NEVER
                          # the v1 namespace, synthetic_vs_real_spoof_probe/, or DETECTOR_RELIABILITY_LOCK_C.json
        defect_corrections:
          defect_a_self_normalizing_threshold: >-
            corrected to a disjoint reference-calibration design: the frozen,
            independent M8 benign population
            (synthesis.quality_calibration.BENIGN_VARIANTS/BENIGN_NOISE_STD)
            is forwarded on the CALIBRATION group only to derive tau_mean/
            tau_tail; the tested corruption is forwarded on the EVALUATION
            group only and scored against that fixed tolerance; delta_plus =
            max(p_after - p_before, 0); no arbitrary constant invented.
          defect_b_cross_route_semantic_mismatch: >-
            cross_route_synthetic reclassified EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL
            -> NEEDS_SCIENTIFIC_DECISION: its canonical declared pass_rule
            ("performance is retained across routes",
            reliability.DECLARED_TESTS) is the opposite direction from the
            BA_sep separability ceiling V1 reused for it; V1's attempt is
            preserved unchanged and unused by V2.
          defect_c_weak_existing_result_validation: >-
            post_failure_diagnostics_v2.validate_existing_diagnostics_result
            now gates every second-or-later --execute and every --status
            call; a complete-but-invalid result set blocks (exit 2) rather
            than being recomputed or overwritten.
          defect_d_placeholder_c8_matrix_identity: >-
            bind_c8_matrix_identity binds the real
            source_matrix.build_plan().identity, cross-checked against
            C8_ACCEPTANCE.json#matrix_identity, fail-closed on mismatch or
            absence, bound at --bind-only and re-verified at --execute.
          defect_e_split_safety_only_global: >-
            verify_per_domain_group_safety fails closed unless BOTH
            casia_fasd and msu_mfsd independently have non-empty, non-
            overlapping calibration/evaluation group sets.
        re_audited_executability:
          benign_jpeg_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          benign_resize_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          benign_color_corruption: {classification: EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL, gpu_ready: true}
          cross_route_synthetic: {classification: NEEDS_SCIENTIFIC_DECISION, gpu_ready: false,
            note: "Defect B: canonical declared meaning is performance retention, not separability; V1's BA_sep-ceiling reuse risked reversing it"}
          recipe_region_shift: {classification: NEEDS_SCIENTIFIC_DECISION, gpu_ready: false,
            note: "unchanged from V1's re-audit"}
          artifact_map_swap: {classification: NEEDS_SCIENTIFIC_DECISION, gpu_ready: false,
            note: "unchanged from V1's re-audit"}
          residual_scale_zero: {classification: STRUCTURALLY_MODEL_BLOCKED, gpu_ready: false,
            note: "unchanged from V1's re-audit"}
          crop_padding_interpolation: {classification: STRUCTURALLY_DATA_BLOCKED, gpu_ready: false,
            note: "unchanged from V1's re-audit"}
        gpu_ready_count: 3   # down from V1's 4 — Defect B correction REMOVES a test
                             # from the executable set rather than inventing a metric
                             # to keep it executable
        benign_corruption_threshold: "reference_mean_delta_plus + 3*reference_std_delta_plus (tau_mean) AND reference_p95_delta_plus + 3*reference_std_delta_plus (tau_tail), both required; reference = frozen M8 BENIGN_VARIANTS/BENIGN_NOISE_STD on the calibration group only, disjoint from the tested corruption"
        cross_route_reuses_ba_sep_ceiling: false   # Defect B: no longer executed under this name
        checkpoint_policy_identical_to_ba_sep: true   # 5/arm, 15 total, same seeds, unchanged
        c9_may_close_hard_coded_false_in_every_artifact: true
        runner_dry_run_on_this_laptop:
          preflight_only: {exit_code: 2, target_access: 0, blocked_on: "missing pydantic-backed source inputs; missing GPU checkpoints; missing reports/full/c8/C8_ACCEPTANCE.json"}
          bind_only: {exit_code: 2, artifacts_written: false, reason: not attempted — preflight already blocked}
          status: {exit_code: 2, diagnostics_result_available: false}
          execute: {exit_code: 2, executed: false, reason: no diagnostics binding on disk}
        real_diagnostic_value_computed: false
        ba_sep_artifacts_untouched_confirmed_by_regression: true
        v1_namespace_untouched_confirmed_by_regression: true
        record: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_PREEXECUTION_CORRECTION.md
        record_json: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_PREEXECUTION_CORRECTION.json
        gpu_handoff: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_GPU_HANDOFF.md
        tests:
          new_file: tests/pipeline/test_c9_post_failure_source_diagnostics_v2.py   # 59 passed
          v1_suite_rerun_unmodified: 70 passed   # proof V1's observable behavior is untouched
          all_c9_scoped_tests_total: 346
      still_unresolved:
        - the other eight REQUIRED_DETECTOR_RELIABILITY_TESTS (residual_scale_zero,
          recipe_region_shift, artifact_map_swap, cross_route_synthetic,
          benign_jpeg_corruption, benign_resize_corruption, benign_color_corruption,
          crop_padding_interpolation) — under the CURRENT (V2) protocol, 3 now
          EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL (the three benign-corruption
          tests; not yet run), 3 NEEDS_SCIENTIFIC_DECISION (recipe_region_shift,
          artifact_map_swap, cross_route_synthetic — the last reclassified
          from V1's EXECUTABLE by the Defect B correction), 1
          STRUCTURALLY_MODEL_BLOCKED, 1 STRUCTURALLY_DATA_BLOCKED
        - crop_padding_interpolation's structural data-block (not reclassified)
        - GPU execution of post_failure_source_diagnostics_v2's 3 executable tests
          (protocol frozen, implementation ready, never run — see V2 GPU handoff doc)
        - cross_route_synthetic, recipe_region_shift, artifact_map_swap remain
          NEEDS_SCIENTIFIC_DECISION pending a normative audit or explicit user
          decision that freezes a faithful metric + threshold for each
      resolved:
        - synthetic_vs_real_spoof_probe = FAILED (scientifically run, observed,
          user-reported; barrier registration also user-reported complete —
          see ba_sep_option1_v2_gpu_registration; not yet laptop-independently verified)
      may_not_be_resolved_from: C8 outcomes or any reliability metric
      c9_may_not_close_until: >-
        every required reliability test genuinely resolves PASSED under its
        own frozen protocol and evidence, and a real, executed
        DETECTOR_RELIABILITY_LOCK_C exists with overall=PASSED. UNDER THE
        CURRENT PROTOCOL VERSION THIS IS NOW UNREACHABLE:
        synthetic_vs_real_spoof_probe has genuinely FAILED, and
        barrier_state's overall=FAILED is sticky — resolving the other eight
        tests to PASSED cannot change overall back to PASSED. Reopening a
        path to target requires either accepting Option A (closure) or a
        NEW, separately versioned protocol (Option C in
        post_ba_failure_decision_dossier); it does not require, and cannot
        be achieved by, running the remaining eight tests under the current
        protocol version.
      no_fake_or_hardcoded_pass_lock_created: true
    c10_c13:
      status: NOT_YET_SCIENTIFICALLY_EXECUTED
      fixture_leak_guards: IMPLEMENTED

  # --- target firewall --------------------------------------------------------
  target_firewall:
    target_access_through_c8: 0
    c8_scope: SOURCE_ONLY
    c9_scope: SOURCE_ONLY   # SOURCE_MATRIX_LOCK_C freezes source-side evidence; no target
    target_must_remain_inaccessible_during_c8_and_c9: true
    target_guided_tuning_authorized: false
    target_evaluation_occurred: false
    note: >-
      C8 trained and selected on source_dev only; cross-source is diagnostic.
      A P3-ready row's held-out evaluation is deferred to C11, entirely
      outside C8. The C9 detector-reliability decision audit (this milestone)
      opened no target root, path or label either. No target root, path or
      label has been resolved anywhere in this reconciliation.

  # --- detector reliability / BA_sep sequencing --------------------------------
  detector_reliability_sequencing:
    order: POST_C8_PRE_C9
    moved_before_c8: false
    stage_constant: C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C
    lock: reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
    lock_exists: false
    # synthetic_vs_real_spoof_probe's PROTOCOL is now frozen as V2 (Option 1,
    # user-approved, pre-execution group-identity correction over V1) — see
    # stage_table.c9.ba_sep_option1_v2_preexecution_correction above. The
    # other eight required tests' protocols remain unresolved, and no test's
    # RESULT (including this one) has been executed under either version.
    unresolved_decisions_remaining: [DETECTOR_BA_SEP_EVIDENCE_VECTOR_NEEDS_SCIENTIFIC_DECISION,
                                     DETECTOR_BA_SEP_PROBE_SEEDS_NEEDS_SCIENTIFIC_DECISION]
    resolved_decisions:
      - id: DETECTOR_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION
        resolved_for: synthetic_vs_real_spoof_probe only (one of nine required tests)
        resolved_by: C9_DETECTOR_BA_SEP_OPTION1_V2, user-approved 2026-08-26 (supersedes V1, same date)
        note: >-
          the EVIDENCE_VECTOR and PROBE_SEEDS sub-questions are answered
          WITHIN this one frozen protocol (Option 1 / the reused seed
          triple), unchanged by the V2 correction; the two decision codes
          above stay listed as "remaining" because they are named at the
          barrier level and the other eight tests do not share this
          protocol.
    decision_dossier: reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md
    protocol_freeze_record: reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.md   # V1, historical
    protocol_correction_record: reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md   # V2, current
    runner_integration_fix_record: reports/readiness/C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.md   # joint bind + real execute
    inference_parity_fix_record: reports/readiness/C9_BA_SEP_OPTION1_V2_INFERENCE_PARITY_FIX.md   # device/CUDA/batching parity
    scientific_failure_closure_record: reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.md   # the observed FAIL
    post_failure_decision_dossier: reports/readiness/C9_DETECTOR_RELIABILITY_POST_BA_FAILURE_DECISION_DOSSIER.md
    post_failure_source_diagnostics_v1_record: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_PROTOCOL_FREEZE.md   # historical, superseded before execution
    post_failure_source_diagnostics_v1_gpu_handoff: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_GPU_HANDOFF.md   # historical, superseded before execution
    post_failure_source_diagnostics_v2_record: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_PREEXECUTION_CORRECTION.md   # current
    post_failure_source_diagnostics_v2_record_json: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_PREEXECUTION_CORRECTION.json
    post_failure_source_diagnostics_v2_gpu_handoff: reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_GPU_HANDOFF.md   # current
    no_ba_sep_number_produced: false   # RESOLVED: synthetic_vs_real_spoof_probe = FAILED, observed
    no_fake_or_hardcoded_pass_lock_created: true

  next_authorized_action: >-
    `synthetic_vs_real_spoof_probe = FAILED` (RND=0.7843079833902619,
    DET=0.8514170182841069, LLM=0.7902658339472685, all over 0.75) is a
    permanent, observed, user-reported scientific result — NOT rerun, tuned,
    or reselected under any circumstance. The GPU host has ALSO
    user-reported registering it into `DETECTOR_RELIABILITY_LOCK_C.json`
    (`identity_sha256 = 40825a5fffcbbdd681e5d8b0354e8371dcccabc36b51d1b9a12cd8bd29e73fbe`,
    `overall=FAILED`, `c9_may_close=false`) — see
    `stage_table.c9.ba_sep_option1_v2_gpu_registration`; this laptop has not
    independently verified those bytes. This milestone found V1
    (`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`, identity
    `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`, NEVER
    scientifically executed) scientifically defective before any GPU run and
    froze a corrected `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2`
    (`configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml`, identity
    `05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523`) —
    bounded, source-only, mechanistic diagnostics that CANNOT reopen C9,
    change BA_sep, or make the barrier PASS. The next authorized action, on
    the GPU host, is the sequence in
    `reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_GPU_HANDOFF.md`
    (NOT the V1 handoff, which is now historical): safe `git fetch` +
    `merge --ff-only` (never `reset --hard`/`clean`/`add -A`, since the GPU
    tree intentionally holds modified/untracked scientific artifacts), a
    protected-state checksum snapshot, `--preflight-only`, then `--bind-only`
    (twice, to prove idempotence — this also binds and cross-checks the real
    `c8_matrix_identity` against `C8_ACCEPTANCE.json`), then `--execute`
    EXACTLY ONCE for the 3 `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` tests
    (`benign_jpeg_corruption`, `benign_resize_corruption`,
    `benign_color_corruption` — `cross_route_synthetic` is no longer among
    them, see Defect B below), then re-running `--execute` and `--status`
    (both now gated by `validate_existing_diagnostics_result`) to prove
    no-rerun, then byte-comparing the BA_sep artifacts,
    `DETECTOR_RELIABILITY_LOCK_C.json`, `C8_ACCEPTANCE.json` and V1's
    (unused) namespace against their pre-diagnostics checksums to prove none
    moved. The remaining 5 required tests stay `NEEDS_SCIENTIFIC_DECISION`
    (`recipe_region_shift`, `artifact_map_swap`, `cross_route_synthetic` —
    the last reclassified from V1's `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL`
    because its canonical declared meaning, "performance is retained across
    routes", is the opposite of the BA_sep separability ceiling V1 reused
    for it) / `STRUCTURALLY_MODEL_BLOCKED` (`residual_scale_zero`) /
    `STRUCTURALLY_DATA_BLOCKED` (`crop_padding_interpolation`) — see
    `stage_table.c9.post_failure_source_diagnostics_v2.re_audited_executability`.
    UNDER THE CURRENT BA_sep PROTOCOL VERSION, NO DIAGNOSTIC OUTCOME CAN
    REOPEN C9: `barrier_state`'s `overall=FAILED` is sticky once one
    required test has genuinely failed, and every diagnostics artifact
    hard-codes `c9_may_close: false` regardless of its own verdict. C7 and
    C8 remain scientifically closed and valid; nothing further is required
    of either. `detector_reliability.verify_lock` remains strict and
    unweakened (`overall == PASSED`-only, confirmed by regression); C9's own
    precondition gate rejects a FAILED reliability record exactly as it
    rejects an absent one (proven by direct test). C9 may not close
    SOURCE_MATRIX_LOCK_C, and C10-C13 may not run, under the current
    protocol version, period. The post-failure EXPLORATORY TARGET protocol
    is explicitly NOT started — a separate, future task, only after these
    diagnostics are observed and frozen. Do not run C9, C10, C11, C12 or
    C13, and do not access target, until then.

last_updated_utc: 2026-08-26   # C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 pre-execution scientific correction of V1 (five defects: self-normalizing benign threshold, cross_route_synthetic semantic mismatch, weak existing-result validation, placeholder C8 matrix identity, split safety proven only globally) — V1 preserved byte-for-byte, unchanged, as historical pre-execution evidence; nothing scientifically executed on the GPU by this task; target_access=0 throughout
```
