# C2C - C3 freeze candidate

**Nothing here is frozen.** This document prepares the exact identities for explicit
user approval. No C3 request was executed.

C2C outcome: **PASS**

## C3 generation contract - candidate components

| component | candidate value |
| --- | --- |
| provider | gemini |
| model | `gemini-3.6-flash` |
| SDK / API surface | google-genai / interactions |
| thinking config | thinking_level = medium, max_output_tokens = 32768, no sampling controls sent |
| system prompt identity | `e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737` |
| batch generation-template identity | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` |
| coverage quota identity | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` |
| single-recipe schema identity | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` |
| batch-envelope schema identity | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` |
| ontology identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| **route policy identity** | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` |
| alias policy | allow_ontology_aliases = False |
| provider config identity | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` |
| request schedule | 12 x 32 = 384 raw slots; min unique pool 320; final bank 256 |
| retry policy | {"semantic_max_retries": 2, "transport_max_attempts": 4, "backoff_initial_seconds": 1.0, "backoff_multiplier": 2.0, "backoff_max_seconds": 60.0} |

## Composite identity

**C3_GENERATION_CONTRACT_IDENTITY**

    884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85

> changing ANY component above changes c3_generation_contract_identity, and therefore invalidates any C3 generation carried out under the previous value.

## Evidence supporting this candidate

| measure | value |
| --- | --- |
| exactly one logical batch | True |
| returned objects | 32 |
| accepted objects | 32 |
| compiler failures among accepted | 0 |
| coverage | coverage preserved |

## What still requires an explicit user decision

1. Approve every identity in the table above as the frozen C3 generation contract.
2. Approve `C3_GENERATION_CONTRACT_IDENTITY` as the value the C3 BANK_LOCK binds to.
3. Confirm the Free-Tier quota position for 12 batch requests before C3 begins; the
   RPM/TPM/RPD limits must be read from AI Studio and were not invented here.

**C3 was not started.**

