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
  branch: v15-spec-reconciliation
  head: 50514cef4b1105bfaf6b3a97c44d1e25588c6448   # branch point; see Git for the reconciliation commit
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
current_substage: pre-generation contract completion
execution_profile: validate
pipeline_phase: preflight
engineering_status: NOT_TESTED      # no v1.5 orchestrator exists yet
scientific_status: NOT_RUN

historical_milestones:
  C0: ACCEPTED_WITH_DOCUMENTED_DEVIATION   # froze against v1.1; v1.5 execution layer is new scope
  C1: ACCEPTED_WITH_DOCUMENTED_DEVIATION   # schema-identity naming corrected prospectively at C2C
  C2: ACCEPTED                             # disposable pilot; never enters C3
  C2B: ACCEPTED_WITH_DOCUMENTED_DEVIATION  # outcome BATCH_SHAPE_FAIL, preserved unchanged
  C2C: ACCEPTED                            # route contract exactly ["physics","gpat"]
  C3_preparation: PARTIAL                  # generation contract only; selection contract absent
  C4..C13: MISSING

c3:
  generation_contract_identity: 884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85
  preliminary_bank_lock_identity: 7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba
  selection_contract_identity: NOT_CREATED
  bank_contract_identity: NOT_CREATED
  c3_scientific_logical_requests: 0
  c3_scientific_candidate_slots: 0
  status: BLOCKED_PENDING_SELECTION_CONTRACT
  supersession: SUPERSESSION_REQUIRED_BEFORE_C3_SCIENTIFIC_GENERATION
  supersession_permitted: true    # §7.8.4, because scientific requests before supersession = 0

execution:
  current_profile: validate
  engineering_status: NOT_TESTED
  scientific_status: NOT_RUN
  pipeline_state: state/PIPELINE_STATE.json      # NOT_CREATED
  master_run_index: state/MASTER_RUN_INDEX.json  # NOT_CREATED
  orchestrator: train.py                          # NOT_CREATED

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
    status: FROZEN, immutable, incomplete under v1.5 §7.8.4

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
  passed: 1310
  failed: 7
  skipped: 101
  milestone_suites: {C0: 32, C1: 138, C2: 43, C2B: 41, C2C: 54, C3: 29}
  inherited_known_failures: 7   # exactly the set documented in reports/c0/C0_TEST_SUITE.json
  new_unexplained_failures: 0

known_deviations:
  - C1 recorded 7afc3abd… as a single-recipe schema identity; it is the 32-object envelope
    identity. Corrected prospectively in docs/c2c/C2C_IDENTITY_CORRECTION_NOTE.md.
  - The C1-recorded bounded 32-object envelope is rejected by the provider (400
    INVALID_ARGUMENT); the working envelope omits the array bound and enforces exactly-32
    on the response.
  - C2B outcome BATCH_SHAPE_FAIL is preserved and must not be rewritten.
  - Spec copied to docs/ per the bootstrap prompt; §M.1 canonical layout says docs/specs/.

blockers:
  - C3 scientific generation is blocked: prism_c3_selection_v1 (§7.8.3) is unimplemented,
    the §7.8.2 coverage quota table is absent, RND/DET 384-slot schedules (§7.8.5) do not
    exist, and no C3_SELECTION_CONTRACT_IDENTITY or C3_BANK_CONTRACT_IDENTITY has been
    computed, so the preliminary lock cannot satisfy §7.8.4.
  - The v1.5 execution layer (profiles, dual status, train.py, state files, run tree) does
    not exist.

next_authorized_action: >
  Implement the v1.5 C3 selection contract — prism_c3_selection_v1 per §7.8.3 with the
  §7.8.2 quota table and the §7.8.5 RND/DET schedules — then compute
  C3_SELECTION_CONTRACT_IDENTITY and C3_BANK_CONTRACT_IDENTITY and prepare a superseding
  C3 BANK_LOCK for user approval, preserving the preliminary lock bytes unchanged. No live
  C3 generation.

last_updated_utc: 2026-08-12
```
