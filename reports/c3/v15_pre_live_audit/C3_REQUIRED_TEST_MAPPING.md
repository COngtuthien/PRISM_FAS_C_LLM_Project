# C3 pre-live gate — required test obligations

Companion to `C3_REQUIRED_TEST_MAPPING.json`. Verification evidence only:
no scientific candidate was generated and no lock was rewritten.

- audit commit: `e850e998ee4387dfd47558452069eff0e848e2df`
- generated: 2026-08-14T07:13:12.253198Z
- focused run: `137 passed in 69.19s (0:01:09)`
- broad run: `7 failed, 1418 passed, 101 skipped, 1 warning in 392.07s (0:06:32)`
- obligations passing: **25 / 25**

A requirement is PASS only when every bound pytest node ID was actually
collected AND reported PASSED in the recorded run. A similarly named file
is not evidence.

| # | Requirement | Tests | Status |
|---|---|---|---|
| 01 | exact 384 candidate slots | 4 | PASS |
| 02 | a pool below 320 eligible fails closed | 2 | PASS |
| 03 | exactly 320 may proceed when feasible | 2 | PASS |
| 04 | final selected cardinality exactly 256 | 3 | PASS |
| 05 | every hard quota holds | 4 | PASS |
| 06 | preferred objective S_pref | 3 | PASS |
| 07 | S_single arithmetic | 3 | PASS |
| 08 | S_multi arithmetic | 2 | PASS |
| 09 | canonical stage-5 tie-break | 6 | PASS |
| 10 | input permutation invariance | 3 | PASS |
| 11 | repeated-run identity | 1 | PASS |
| 12 | filesystem-order invariance | 1 | PASS |
| 13 | solver-order invariance | 2 | PASS |
| 14 | exact-route rejection | 4 | PASS |
| 15 | duplicate handling | 1 | PASS |
| 16 | compiler-invalid exclusion | 1 | PASS |
| 17 | target-field independence | 6 | PASS |
| 18 | synthetic quality q independence | 3 | PASS |
| 19 | downstream detector-score independence | 3 | PASS |
| 20 | 31/33 batch failure preserved | 4 | PASS |
| 21 | no silent truncation or padding | 3 | PASS |
| 22 | RND reproducibility | 3 | PASS |
| 23 | DET reproducibility | 3 | PASS |
| 24 | old preliminary lock integrity | 3 | PASS |
| 25 | zero live Gemini calls | 5 | PASS |

## Bound test node IDs

### 01 — exact 384 candidate slots (PASS)

- `tests/c3/test_c3_schedules_and_identity.py::test_each_control_arm_has_exactly_384_canonical_slot_ids[RND]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_each_control_arm_has_exactly_384_canonical_slot_ids[DET]` → **PASSED**
- `tests/c3/test_c3_selection.py::test_frozen_cardinalities_are_the_spec_values` → **PASSED**
- `tests/c3/test_c3_selection.py::test_the_slot_budget_cannot_be_topped_up` → **PASSED**

### 02 — a pool below 320 eligible fails closed (PASS)

- `tests/c3/test_c3_selection.py::test_a_pool_below_the_minimum_fails_closed` → **PASSED**
- `tests/c3/test_c3_selection.py::test_pool_below_minimum_is_reported` → **PASSED**

### 03 — exactly 320 may proceed when feasible (PASS)

- `tests/c3/test_c3_selection.py::test_exactly_the_minimum_pool_is_permitted_to_proceed` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256` → **PASSED**

### 04 — final selected cardinality exactly 256 (PASS)

- `tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256` → **PASSED**
- `tests/c3/test_c3_selection.py::test_selected_cardinality_is_exact` → **PASSED**
- `tests/c3/test_c3_selection.py::test_frozen_cardinalities_are_the_spec_values` → **PASSED**

### 05 — every hard quota holds (PASS)

- `tests/c3/test_c3_selection.py::test_the_frozen_quota_table_matches_the_spec` → **PASSED**
- `tests/c3/test_c3_selection.py::test_every_hard_quota_holds_in_the_selected_bank` → **PASSED**
- `tests/c3/test_c3_selection.py::test_an_infeasible_hard_quota_fails_rather_than_relaxing` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256` → **PASSED**

### 06 — preferred objective S_pref (PASS)

- `tests/c3/test_c3_selection.py::test_s_pref_is_exact_integer_arithmetic` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf0-8-3]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf1-9-4]` → **PASSED**

### 07 — S_single arithmetic (PASS)

- `tests/c3/test_c3_selection.py::test_s_single_is_exact_integer_arithmetic` → **PASSED**
- `tests/c3/test_c3_selection.py::test_objective_values_are_recomputed_from_the_selected_set` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf2-10-4]` → **PASSED**

### 08 — S_multi arithmetic (PASS)

- `tests/c3/test_c3_selection.py::test_s_multi_is_exact_integer_arithmetic` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf3-11-5]` → **PASSED**

### 09 — canonical stage-5 tie-break (PASS)

- `tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf0-8-3]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf1-9-4]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf2-10-4]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf3-11-5]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_stage_5_decides_alone_when_every_subset_ties` → **PASSED**
- `tests/c3/test_c3_selection.py::test_the_selected_set_is_the_lexicographically_smallest_tied_set` → **PASSED**

### 10 — input permutation invariance (PASS)

- `tests/c3/test_c3_selection.py::test_input_permutation_does_not_change_the_bank` → **PASSED**
- `tests/c3/test_c3_selection.py::test_reversed_input_does_not_change_the_bank` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_tie_case_is_immune_to_input_order` → **PASSED**

### 11 — repeated-run identity (PASS)

- `tests/c3/test_c3_selection.py::test_repeated_runs_are_identical` → **PASSED**

### 12 — filesystem-order invariance (PASS)

- `tests/c3/test_c3_selection_optimality.py::test_filesystem_listing_order_does_not_change_the_bank` → **PASSED**

### 13 — solver-order invariance (PASS)

- `tests/c3/test_c3_selection_optimality.py::test_solver_traversal_order_cannot_decide_the_bank` → **PASSED**
- `tests/c3/test_c3_selection.py::test_candidates_are_ordered_by_canonical_sha_not_by_input_order` → **PASSED**

### 14 — exact-route rejection (PASS)

- `tests/c3/test_c3_selection.py::test_route_invalid_candidates_are_excluded_before_the_compiler` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_declare_the_scientific_route[RND]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_declare_the_scientific_route[DET]` → **PASSED**
- `tests/c3/test_c3_bank_lock.py::test_the_lock_freezes_the_route_contract` → **PASSED**

### 15 — duplicate handling (PASS)

- `tests/c3/test_c3_selection.py::test_duplicates_are_excluded_at_the_deduplication_stage` → **PASSED**

### 16 — compiler-invalid exclusion (PASS)

- `tests/c3/test_c3_selection.py::test_an_uncompilable_candidate_is_excluded` → **PASSED**

### 17 — target-field independence (PASS)

- `tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[siw_mv2_attack_family-print]` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[target_acer-0.0731]` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank` → **PASSED**
- `tests/c3/test_c3_selection.py::test_selection_ignores_target_quality_and_score_fields` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_schedules_carry_no_target_or_llm_dependency[RND]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_schedules_carry_no_target_or_llm_dependency[DET]` → **PASSED**

### 18 — synthetic quality q independence (PASS)

- `tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[synthetic_quality_q-0.421]` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[q-0.421]` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank` → **PASSED**

### 19 — downstream detector-score independence (PASS)

- `tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[detector_score-0.8817]` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_a_forbidden_value_correlated_with_quality_cannot_bias_selection` → **PASSED**
- `tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank` → **PASSED**

### 20 — 31/33 batch failure preserved (PASS)

- `tests/c3/test_c3_selection_optimality.py::test_a_c3_batch_that_is_not_exactly_32_still_fails_closed[31]` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_a_c3_batch_that_is_not_exactly_32_still_fails_closed[33]` → **PASSED**
- `tests/c2b/test_c2b_batch_envelope.py::test_a_batch_that_is_not_exactly_32_is_rejected[31]` → **PASSED**
- `tests/c2b/test_c2b_batch_envelope.py::test_a_batch_that_is_not_exactly_32_is_rejected[33]` → **PASSED**

### 21 — no silent truncation or padding (PASS)

- `tests/c3/test_c3_selection.py::test_no_silent_truncation_or_padding` → **PASSED**
- `tests/c3/test_c3_selection.py::test_a_missing_slot_is_a_rejection_not_an_omission` → **PASSED**
- `tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256` → **PASSED**

### 22 — RND reproducibility (PASS)

- `tests/c3/test_c3_schedules_and_identity.py::test_schedule_identity_is_reproducible[RND]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_candidate_drafting_is_per_slot_deterministic[RND]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_pass_the_common_eligibility_gate[RND]` → **PASSED**

### 23 — DET reproducibility (PASS)

- `tests/c3/test_c3_schedules_and_identity.py::test_schedule_identity_is_reproducible[DET]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_candidate_drafting_is_per_slot_deterministic[DET]` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_pass_the_common_eligibility_gate[DET]` → **PASSED**

### 24 — old preliminary lock integrity (PASS)

- `tests/c3/test_c3_bank_contract_lock.py::test_the_historical_preliminary_lock_is_byte_identical` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_the_preliminary_lock_bytes_are_unchanged` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_the_preliminary_lock_still_binds_no_selection_identity` → **PASSED**

### 25 — zero live Gemini calls (PASS)

- `tests/c3/test_c3_schedules_and_identity.py::test_the_ambient_credential_is_removed` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_importing_the_c3_selection_path_loads_no_provider_sdk` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_selection_and_schedule_modules_import_no_provider` → **PASSED**
- `tests/c3/test_c3_schedules_and_identity.py::test_no_c3_scientific_generation_artifact_exists` → **PASSED**
- `tests/c3/test_c3_bank_lock.py::test_no_c3_scientific_request_has_been_made` → **PASSED**

