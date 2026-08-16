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
  branch: c3-v15-selection-contract
  branch_point: a876d5ffba410a867173c7ca719618b2b48a5144   # the accepted v1.5 reconciliation commit
  latest_accepted_checkpoint: c2c84aecc7e6fce84a18f5dc3ab32d531feed2c5   # C2C = PASS
  origin: https://github.com/COngtuthien/PRISM_FAS_C_LLM_Project.git
  version_b_remote_push: DISABLED_NO_PUSH_TO_VERSION_B

version_b:
  repo_path: D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project
  head: 7799f7decd35db6987ce4578824e5bd8d9eab4ae
  tag: m10-blind-evaluation-checkpoint
  tag_peeled_commit: 7799f7decd35db6987ce4578824e5bd8d9eab4ae
  clean: true
  immutable_verified: true

current_milestone: C3
current_substage: v1.5 execution skeleton (restructure step 2) — COMPLETE
execution_profile: validate
pipeline_phase: preflight

# SCOPE WARNING. `engineering_status` below describes ONLY the C3 selection/bank
# contract and its offline test suite. The execution layer now exists and its
# validate profile has been run (see `execution_layer`), but a validate PASS means
# the readiness checks passed — NOT that any stage executed, NOT that smoke or full
# can run, and NOT that any milestone is scientifically complete.
engineering_status_scope: c3_selection_contract_substage
engineering_status: SMOKE_PASS      # contract + selector frozen and tested offline
scientific_status: NOT_RUN          # no C3 scientific generation has occurred

# Machine-readable so no parser can infer these exist. Previously this fact lived
# only in YAML comments beside real-looking paths, which a parser strips.
execution_pipeline:
  # The validate profile is implemented and has been executed. It performs the
  # L.2 readiness checks only — identities, locks, contracts, environment — and
  # executes NO stage. smoke and full require stage adapters, which do not exist.
  validate: IMPLEMENTED
  smoke: NOT_IMPLEMENTED        # refuses to run: BLOCKED, exit 2
  full: NOT_IMPLEMENTED         # refuses to run: BLOCKED, exit 2
  orchestrator_exists: true     # train.py + src/prism_fas/pipeline/
  pipeline_state_exists: true   # state/PIPELINE_STATE.json
  master_run_index_exists: true # state/MASTER_RUN_INDEX.json
  stage_adapters_exist: false   # all 14 stages declare adapter_implemented=false
  c0_to_c13_pipeline_ever_run: false
  note: >-
    No stage has ever been EXECUTED. `validate: IMPLEMENTED` means the readiness
    checks run and pass; it does not mean any milestone can execute. Under the smoke
    or full profile the orchestrator stops with BLOCKED and names the missing
    adapters. Nothing here is scientific evidence.

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
  C4..C13: MISSING

c3:
  selection_implementation: COMPLETE        # prism_c3_selection_v1, §7.8.3
  contract_supersession: COMPLETE           # §7.8.4 superseding pre-scientific lock written
  scientific_generation: NOT_RUN            # C3 itself is NOT scientifically complete

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

  c3_scientific_logical_requests: 0
  c3_scientific_candidate_slots: 0
  raw_candidate_archives_created: 0
  recipe_banks_selected: 0
  status: CONTRACT_FROZEN_AWAITING_USER_APPROVAL_BEFORE_SCIENTIFIC_GENERATION

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
  current_profile: validate
  engineering_status: NOT_TESTED   # the EXECUTION LAYER, not the C3 contract substage
  scientific_status: NOT_RUN
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
  - path: reports/c3/C3_BANK_LOCK.json
    identity: 7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba
    status: FROZEN, immutable, SUPERSEDED — historical evidence, bytes preserved unchanged
  - path: reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json
    identity: 1acdf68f56195f1b568449b545865ae2868d99d480ed6b75b28215178c5e9628
    status: PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK — AWAITING USER APPROVAL
    note: >-
      A CONTRACT lock. It does not assert that any per-arm 384-slot raw archive or
      256-recipe bank exists; none has been generated or selected.

historical_live_provider_calls:
  total_attempts_before_c3: 47      # 46 archived + 1 documented-but-overwritten (C2B 400)
  semantic_responses: 36
  c1: 0
  c2_smoke: 2
  c2_pilot: 42
  c2b: 1 archived (+1 documented in C2B_ENVELOPE_REJECTION.json)
  c2c: 1
  c3_scientific: 0

tests:
  latest_exact_command: python -m pytest -q --no-header -p no:cacheprovider --continue-on-collection-errors
  passed: 1568
  failed: 7
  skipped: 101
  milestone_suites: {C0: 32, C1: 138, C2: 43, C2B: 41, C2C: 54, C3: 156, pipeline: 131}
  pipeline_suite_exact_command: python -m pytest tests/pipeline -q --no-header -p no:cacheprovider
  pipeline_suite: {passed: 131, failed: 0, skipped: 0}
  pipeline_offline: >-
    sockets blocked and ambient credentials deleted by an autouse fixture; a static
    AST check asserts that no pipeline module and no train.py path imports a provider,
    Modal, torch or target-evaluation module
  inherited_known_failures: 7   # exactly the set documented in reports/c0/C0_TEST_SUITE.json
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
  - C3 scientific generation remains blocked on USER APPROVAL of the superseding
    bank-contract lock. The §7.8.3 selector, the §7.8.2 quota table and the §7.8.5
    RND/DET schedules now exist and are identity-bound; what is missing is the decision,
    not the contract.
  - Stage adapters do not exist. All 14 stages declare adapter_implemented=false, so the
    smoke and full profiles stop with BLOCKED. This is restructure-plan step 4 and is
    separately authorized. Until it lands, no milestone can be EXECUTED by train.py.
  - Steps 3, 5 and 6 of the restructure plan remain: dual-status stamping of newly written
    artifacts beyond the pipeline's own, the Appendix M.1 layout move with identity
    re-derivation, and the C4..C13 stages themselves.
  - Quota snapshot for gemini-3.6-flash (RPM/TPM/RPD) is still NOT_PROGRAMMATICALLY_AVAILABLE;
    the manual AI Studio step recorded in reports/c3/C3_BANK_LOCK.json is still outstanding.

deviations_recorded_this_session:
  - The v1.5 execution layer was built ahead of the recorded next_authorized_action. The
    user authorized it explicitly in session; the C3 review it displaced is unchanged and
    is restored as the next authorized action below. No C3 artifact, lock or identity was
    touched, and the validate run re-derived all of them unchanged.
  - L.3 provides no engineering-status value meaning "validate passed". Recorded as an
    interpretation rather than silently reconciled: engineering_status stays NOT_TESTED
    and the run reports on a separate validate_gate axis. See execution_layer above.
  - .gitignore re-includes reports/{validate,smoke,full}/ so execution evidence is
    committable, matching the existing rationale for the reports/c0..c13 roots.

next_authorized_action: >
  USER REVIEW of the C3 pre-live audit (reports/c3/v15_pre_live_audit/) and the superseding
  bank-contract lock, before any live Gemini 12x32 C3 scientific generation. The pre-live
  audit gate is verification evidence only; it does not authorize generation. Unchanged by
  this session: the execution layer neither advanced nor weakened the C3 decision, and the
  validate run confirmed all C3 identities and locks still reproduce.

last_updated_utc: 2026-08-16
```
