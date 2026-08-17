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

current_milestone: PREPARATION_PIPELINE_COVERAGE_CLOSURE
current_substage: derived-data preparation unit/integration coverage, and the three
                  production defects it exposed — COMPLETE
previous_milestone: FINAL_FULL_PATH_CUDA_PORTABILITY_ONE_FOLDER_DATA_CLOSURE
execution_profile: rehearsal   # `python train.py` resolved CPU_FULL_REHEARSAL here
pipeline_phase: engineering-readiness

# SCOPE WARNING. The two axes below say different things and are easy to conflate.
#
# `engineering_status: SMOKE_PASS` is now scoped to the WHOLE C0-C13 pipeline: every
# stage has an adapter, and `--profile smoke --from C0 --to C13` traverses all
# fourteen and passes. That is an ENGINEERING statement about code paths.
#
# `scientific_status` is unchanged and remains scoped to C3. C4-C13 now have
# adapters and STILL have never executed scientifically. An adapter proves a stage
# CAN run; it is not evidence that it HAS. Read `execution_pipeline` and
# `historical_milestones` before quoting either field.
engineering_status_scope: C0_TO_C13_FULL_PIPELINE_ENGINEERING
engineering_status: SMOKE_PASS      # C0-C13, validate PASS + smoke PASS, 2026-08-17
scientific_status: PASS             # C3 ONLY. C4-C13 = NOT_RUN.
scientific_status_scope: c3_only

# Machine-readable so no parser can infer these exist. Previously this fact lived
# only in YAML comments beside real-looking paths, which a parser strips.
execution_pipeline:
  validate: IMPLEMENTED_C0_TO_C13   # 14 stages, 21 checks, 0 failed
  smoke: IMPLEMENTED_C0_TO_C13      # 14 stages, 62 substage modes, 242 checks, 0 failed
  full: EXECUTED_C3_ONLY            # live C3 generation ran 2026-08-16; C4-C13 BLOCKED
  # Separate axis, deliberately not folded into `full` above. The scientific CODE
  # PATH for C4-C13 now exists; that is not a claim that it ran. The refusing
  # `run_full` is GONE — one `workflow()` per stage, driven by an ExecutionContext,
  # with no placeholder to fall back to. On this laptop every stage still BLOCKS on
  # a named missing input, never on missing code.
  full_path_implemented_c4_to_c13: true
  full_path_placeholders_remaining: 0
  full_path_ever_executed_c4_to_c13: false
  orchestrator_exists: true     # train.py + src/prism_fas/pipeline/
  pipeline_state_exists: true   # state/PIPELINE_STATE.json
  master_run_index_exists: true # state/MASTER_RUN_INDEX.json
  stage_adapters_exist: C0_TO_C13_ALL
  c0_to_c13_pipeline_ever_run: ENGINEERING_ONLY
  c0_to_c13_pipeline_ever_run_scientifically: false
  note: >-
    Every C0-C13 stage has an adapter and every stage's control path executes and
    passes under validate and smoke. C3 additionally executed LIVE under --profile
    full and is scientifically complete. C4-C13 have executed ONLY on fixtures:
    under --profile full each of them BLOCKS on an absent scientific input and names
    it, and no scientific training, synthesis, target inference or scoring has ever
    run. SMOKE_PASS over C0-C13 means ENGINEERING_READY, not scientific completion.

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
    substage_modes: 62
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

# --- preparation pipeline coverage closure (this milestone) ------------------
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
  passed: 1927
  failed: 7
  skipped: 101
  measured_at_utc: 2026-08-17   # after the preparation coverage closure
  milestone_suites: {C0: 32, C1: 138, C2: 43, C2B: 41, C2C: 54, C3: 156, C7: 19,
                     pipeline: 313}
  new_tests_this_milestone: 97   # 26 search engine + 35 portability/contracts +
                                 # 19 decision contract + 17 pipeline
  pipeline_suite_exact_command: python -m pytest tests/pipeline -q --no-header -p no:cacheprovider
  pipeline_suite: {passed: 471, failed: 0, skipped: 0}
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

next_authorized_action: >
  USER REVIEW before copying the complete portable bundle to an external NVIDIA CUDA
  machine and running exactly `python train.py`. Nothing else is authorized.

  What review is being asked to confirm: C4-C13 now have a real production FULL code
  path with zero placeholders; C8 under full schedules all 42 declared rows of the
  frozen matrix a777671f... and cannot see the rehearsal sampler; a rehearsal cannot
  serve as a scientific ancestor through any of four independent barriers; the runner
  grades any NVIDIA host on compute capability and driver rather than on model name;
  and the folder needs no pip command, no dataset-preparation command and no
  train.py argument.

  What review must NOT read into it: no CUDA hardware has ever been validated, no
  reports/full/c4..c13 artifact has ever been written, the production output audit
  is structural rather than observed, and derived-data preparation has been
  exercised only against stub builders — never against the real corpora. The first
  external-GPU run is therefore the first execution of that path, and it should be
  watched rather than left alone.

  What changed since the last review: the preparation coverage gap is closed, and
  closing it found three real defects — a TypeError that would have aborted every
  genuine M3A build, a paths config that could not survive the folder being copied,
  and an interrupted package that looked complete to C4. All three would have
  fired on the collaborator's machine rather than here.

  No C4-C13 scientific execution, GPU allocation, Modal spend, target access or
  Gemini call has been performed or is authorized.

last_updated_utc: 2026-08-17   # final full-path / CUDA-portability / one-folder closure
```
