# E8 Training Runner V2.2 -- Recipe-Binding-Corrected Preflight

**Classification:** additive integration/runtime-input binding correction only. No GPU, no training,
no smoke, no target access, no LLM call, no commit/push.

**Status: `READY_FOR_E8_RUNNER_V2_2_COMMIT_REVIEW`**

## The two recipe-bank contracts

- **Contract A (C3 treatment bank):** arm-specific, per-arm identity, feeds `C6MatchedBankReader`.
- **Contract B (M7 neutral bank):** shared across all arms, feeds `M9Trainer.recipe_bank_root`.

Resolved by the new `resolve_e8_runtime_inputs(spec, root)` helper, which delegates entirely to
`prism_fas.pipeline.adapters.sources.verify_detector_inputs` and
`prism_fas.synthesis.c5_arm_plan.load_arm_bank` -- never re-implements either resolver's validation.

## Per-arm C3 treatment bank (resolved locally in this checkout)

```
{
  "RND": {
    "arm": "RND",
    "root": "assets/recipe_banks/c3/rnd",
    "available": true,
    "error": null,
    "identity": "07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136",
    "expected_identity": "07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136",
    "identity_matches_expected": true,
    "recipe_count": 256,
    "expected_recipe_count": 256
  },
  "DET": {
    "arm": "DET",
    "root": "assets/recipe_banks/c3/det",
    "available": true,
    "error": null,
    "identity": "2802ca5f537c4278eefdb160049d52cb1b667234ec5e32736a733b272e9231c9",
    "expected_identity": "2802ca5f537c4278eefdb160049d52cb1b667234ec5e32736a733b272e9231c9",
    "identity_matches_expected": true,
    "recipe_count": 256,
    "expected_recipe_count": 256
  },
  "LLM": {
    "arm": "LLM",
    "root": "assets/recipe_banks/c3/llm",
    "available": true,
    "error": null,
    "identity": "f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59",
    "expected_identity": "f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59",
    "identity_matches_expected": true,
    "recipe_count": 256,
    "expected_recipe_count": 256
  }
}
```

## Ordering guarantee

`verify_source_binding_correction() -> canonical detector input verification (verify_detector_inputs) -> M3B+M7 identity checks -> target firewall zero-check -> C3 arm-bank identity/count check -> E8 membership + C6 bank opening -> frozen Track-G config resolution -> collision guard -- confirmed by test_v21_launch_verifies_correction_identity_before_m3b_and_trainer and the V2.2 test_v22_23/24 failure-order tests`

## Frozen contract -- unchanged by this correction

- 15 run IDs: unchanged (15 total)
- Seeds: `[20260806, 20260807, 20260808, 20260809, 20260810]`
- M3B content identity: `08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9`
- E7-D source-support identity: `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b`
- M7 detector recipe bank: `{'root': 'assets/recipe_banks/prism_recipe_bank_m7_v1', 'identity': 'fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb', 'recipe_count': 128}`
- C3 treatment bank roots/identities by arm: `{'root_by_arm': {'DET': 'assets/recipe_banks/c3/det', 'LLM': 'assets/recipe_banks/c3/llm', 'RND': 'assets/recipe_banks/c3/rnd'}, 'identity_by_arm': {'DET': '2802ca5f537c4278eefdb160049d52cb1b667234ec5e32736a733b272e9231c9', 'LLM': 'f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59', 'RND': '07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136'}, 'expected_recipe_count': 256}`
- C5 candidates root: `runs/full/c5/scientific/candidates`
- Bank counts: `{'total': 818, 'physics': 354, 'gpat': 464}`
- Schedule: `{'total_optimizer_updates': 1575, 'synthetic_draws_per_run': 10800}`
- Target firewall: `{'target_access': False, 'target_labels_accessed': False}`
- `configs/models/m9_detector.yaml`: **not modified**

## Focused tests

```
pytest -q tests/test_c_ext_e8_training_runner.py
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py tests/test_m4_loader_sampler.py
```

Combined: **187 passed, 0 failed, 16 skipped**. Runner suite alone: **119 passed, 0 failed, 0
skipped** (84 pre-existing + 35 new V2.2 tests). No broad `-k c_ext` sweep.

## Historical integrity

V1 execution plan + runner reports unchanged; V2 and V2.1 correction reports unchanged; membership,
adapter, historical C6, E7 unchanged; C3 banks and M7 bank unchanged; `configs/models/m9_detector.yaml`
unchanged.

## Identities

- New runner rule name: `E8_FIXED_TRACK_G_RUNNER_V2_2_RECIPE_BINDING_FIXED`
- New runner rule identity: `1e4a66fb2f49d5ad7b512ab01293fa8dd66e9b5ccaa75af3fa95b2a99689130b`
- New runner source SHA256: `60a3c72cbf425f4ab71aacc5a08cfcb831779308517c952db1c44d5501a686ea`
- Implementation parent commit: `909bb77962503ff83a85e6e0ee509fdde24af47a`

## Status

**`READY_FOR_E8_RUNNER_V2_2_COMMIT_REVIEW`**
