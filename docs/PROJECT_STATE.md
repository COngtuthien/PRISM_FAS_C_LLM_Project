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

current_milestone: PORTABLE_ONE_COMMAND_EXECUTION_CLOSURE
current_substage: portable one-folder runner, bootstrap and reporting — COMPLETE
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
    resolved_intent: CPU_FULL_REHEARSAL
    outcome: PASS
    stages: 14
    engineering_status_per_stage: SMOKE_PASS
    scientific_status_per_stage: NOT_RUN
    reports_namespace: reports/rehearsal
    runs_namespace: runs/rehearsal
    plots_written: 4
    tables_written: 9
    report_html: reports/rehearsal/final/report.html
    scientific_outputs_modified: 0
    real_target_access: 0

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
  passed: 1769
  failed: 7
  skipped: 101
  milestone_suites: {C0: 32, C1: 138, C2: 43, C2B: 41, C2C: 54, C3: 156, C7: 19,
                     pipeline: 313}
  new_tests_this_milestone: 97   # 26 search engine + 35 portability/contracts +
                                 # 19 decision contract + 17 pipeline
  pipeline_suite_exact_command: python -m pytest tests/pipeline -q --no-header -p no:cacheprovider
  pipeline_suite: {passed: 313, failed: 0, skipped: 0}
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
    under --profile full on this machine and names the input it lacks."
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
  USER REVIEW of the portable pre-full checkpoint before optionally testing
  `python train.py` on the local CPU laptop or transferring the complete folder to the
  external CUDA GPU. The runner resolves CPU_FULL_REHEARSAL here and passes end to end;
  on a compatible CUDA host the same command resolves GPU_SCIENTIFIC_FULL starting at C4.
  No C4-C13 scientific execution, GPU allocation, Modal spend, target access or Gemini
  call has been performed or is authorized.

last_updated_utc: 2026-08-17   # portable one-command execution closure
```
