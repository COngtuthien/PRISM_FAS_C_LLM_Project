# C3 generation BANK_LOCK

**Status: FROZEN.** Frozen before any C3 scientific request. This file records
the contract every C3 request must run under; it contains no recipe.

- `bank_lock_identity`: `7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba`
- **`C3_GENERATION_CONTRACT_IDENTITY`**: `884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85`

## Frozen components

| component | frozen value |
| --- | --- |
| provider | gemini |
| model | `gemini-3.6-flash` |
| SDK / API surface | google-genai / interactions |
| thinking | thinking_level = medium, max_output_tokens = 32768, no sampling controls |
| system prompt identity | `e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737` |
| batch generation-template identity | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` |
| coverage quota identity | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` |
| single-recipe schema identity | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` |
| batch-envelope schema identity | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` |
| ontology identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| route policy identity | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` |
| alias policy | allow_ontology_aliases = False |
| provider config identity | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` |
| retry policy | `{'semantic_max_retries': 2, 'transport_max_attempts': 4, 'backoff_initial_seconds': 1.0, 'backoff_multiplier': 2.0, 'backoff_max_seconds': 60.0}` |
| request schedule | 12 x 32 = 384 raw slots; min unique pool 320; final bank 256 |

## Composite identity

The composite is taken over exactly these keys:

    allow_ontology_aliases, api_surface, batch_envelope_schema_identity, batch_generation_template_identity, coverage_quota_identity, max_output_tokens, model_id, ontology_identity, provider, provider_config_identity, request_schedule, response_mime_type, retry_policy, route_policy_identity, sdk_package, single_recipe_schema_identity, system_prompt_identity, thinking_level

Canonical form: `json.dumps(components, sort_keys=True, separators=(',',':'), ensure_ascii=False) then SHA-256 over the UTF-8 bytes`

> changing ANY component changes this identity and invalidates any C3 generation carried out under the previous value

## Verification against the approval

Every identity below was **re-derived from the code and configuration on disk**, not
copied from the approval text.

| component | approved | re-derived | matches |
| --- | --- | --- | --- |
| provider | `gemini` | `gemini` | yes |
| model_id | `gemini-3.6-flash` | `gemini-3.6-flash` | yes |
| api_surface | `interactions` | `interactions` | yes |
| sdk_package | `google-genai` | `google-genai` | yes |
| thinking_level | `medium` | `medium` | yes |
| response_mime_type | `application/json` | `application/json` | yes |
| max_output_tokens | `32768` | `32768` | yes |
| system_prompt_identity | `e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737` | `e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737` | yes |
| batch_generation_template_identity | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` | yes |
| coverage_quota_identity | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` | yes |
| single_recipe_schema_identity | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` | yes |
| batch_envelope_schema_identity | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` | yes |
| ontology_identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` | yes |
| route_policy_identity | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` | yes |
| allow_ontology_aliases | `False` | `False` | yes |
| provider_config_identity | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` | yes |
| request_schedule | `{'requests': 12, 'objects_per_request': 32, 'raw_slots': 384, 'minimum_unique_pool': 320, 'final_bank': 256}` | `{'requests': 12, 'objects_per_request': 32, 'raw_slots': 384, 'minimum_unique_pool': 320, 'final_bank': 256}` | yes |
| retry_policy | `{'semantic_max_retries': 2, 'transport_max_attempts': 4, 'backoff_initial_seconds': 1.0, 'backoff_multiplier': 2.0, 'backoff_max_seconds': 60.0}` | `{'semantic_max_retries': 2, 'transport_max_attempts': 4, 'backoff_initial_seconds': 1.0, 'backoff_multiplier': 2.0, 'backoff_max_seconds': 60.0}` | yes |

- lock body hash reproducible: **True**
- composite reproducible from code: **True**
- composite equals the approved value: **True**
- problems: **none**

## Route contract

`generator_route` must be exactly `["physics", "gpat"]`. Physics-only and gpat-only are
both rejected; there is no GPAT-only accepted class; silent repair is never permitted.

## Request schedule

exactly 12 logical requests of exactly 32 recipe objects; a response whose recipe count is not exactly 32 fails closed

- minimum valid unique pool before selection: **320** —
  C3 FAILS; the validator is never weakened after seeing results
- final bank: **256**
- selection: deterministic and algorithmic; no manual cherry-picking of LLM recipes

## Free-Tier operating policy (approved)

- Free Tier only. Code never enables billing.
- Transient short-window 429: retry the exact frozen request under the approved bounded
  backoff.
- True daily/project quota exhaustion: checkpoint completed scientific requests and stop
  cleanly.
- Quota never changes the provider, model, prompt, schema, ontology, quotas, route policy,
  request schedule or this lock. A resumed run uses the same frozen contract.

## Quota snapshot

NOT_PROGRAMMATICALLY_AVAILABLE — The active project's RPM/TPM/RPD limits are shown only in the AI Studio web console, which requires an interactive authenticated session this process does not have. No value was invented.

Manual step before C3: Open AI Studio for the project owning GEMINI_API_KEY and record the RPM / TPM / RPD shown for gemini-3.6-flash on the current tier.

## Prohibited during C3

no GPU training, no GPAT training, no synthetic image generation, no detector training, no SiW label access, no SiW metric use, no target scoring, no prompt change, no schema change, no ontology change, no quota change, no route policy change, no model or provider change, no automatic billing.

## Immutability

this file is written once. A later build that would produce different bytes raises rather than overwriting, so a generation run can always be traced to the contract it ran under.
