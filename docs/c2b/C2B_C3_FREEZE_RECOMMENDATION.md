# C2B - C3 freeze recommendation

**This document does not freeze anything.** It prepares the exact candidate values
for user approval. The C3 prompt/request contract is frozen only by an explicit user
decision.

C2B outcome: **BATCH_SHAPE_FAIL**

## Candidate identities awaiting user approval

| field | candidate value |
| --- | --- |
| provider | Google Gemini Developer API |
| model | `gemini-3.6-flash` |
| API surface | interactions |
| thinking_level | medium |
| system prompt identity | `d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f` |
| batch generation-template identity | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` |
| coverage quota identity | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` |
| single-recipe schema identity | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` |
| batch-envelope schema identity | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` |
| ontology identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| alias policy | allow_ontology_aliases = False |
| request schedule | 12 requests x 32 recipes = 384 raw candidate slots |

## What changed since the C1 freeze, and why

1. **The batch envelope had to change.** The 32-object envelope C1 recorded
   (`7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4`)
   carries `minItems = maxItems = 32` and the provider rejects it outright with
   `400 INVALID_ARGUMENT`. C3 as recorded at C1 could not have run. The envelope
   C2B sends is the same schema without that bound; the requirement for exactly 32
   objects is unchanged and is enforced locally on the response.
2. **The single-recipe item schema did not change.** It is byte-identical in the
   1-object envelope C2 used and in the envelope C2B sends, which is the evidence
   that recipe semantics were untouched.
3. **A batch generation template and coverage quotas were added.** Both are
   ontology-level and source-independent. The system instruction was not edited.

## Blocking issue found: valid recipes the compiler cannot build

10 of 32 accepted
recipes declared `generator_route` without `physics`. Those recipes are fully valid -
the ontology enables both routes and the validator only checks membership - but
`compile_recipe` refuses them, because an operator graph is a physics-route artifact.
The validator and the compiler disagree about what an acceptable recipe is.

Route distribution: `{"gpat": 10, "physics": 16, "physics+gpat": 6}`

- Every one of the 32 C2 singleton recipes declared the physics route (25 physics+gpat, 7 physics-only). The batch request is the first context in which the model chose gpat-only.
- Caused by the coverage quotas: **no**. The C2B quotas constrain media, geometry, illumination, artifacts and regions. They never mention generator_route, so they cannot have forced this choice.

**This must be decided before C3.** Two options, both a user decision:

1. Require the physics route (in the system instruction, the item schema, or the
   validator). This changes the single-recipe schema identity if done in the schema.
2. Accept gpat-only recipes as a distinct class that never enters the operator-graph
   compiler, and decide what the synthesis path does with them.

C2B changed neither. It only measured the disagreement.

## Coverage evidence behind the recommendation

| axis | C2 singleton present | C2B batch present | collapse resolved |
| --- | --- | --- | --- |
| artifacts | 6/8 | 8/8 | yes |
| regions | 8/9 | 9/9 | yes |
| media | 2/5 | 5/5 | yes |
| geometry | 2/6 | 6/6 | yes |
| illumination | 2/6 | 6/6 | yes |

Required quota failures: 0. Preferred misses: 0.

## Corrected C3 cost estimate

| measure | value |
| --- | --- |
| design | 12 requests x 32 objects = 384 raw slots |
| expected batch calls (minimum) | 12 |
| expected transport retry burden | 0 |
| expected valid candidates | 384.0 |
| serial model time (minutes, range) | 7.6 - 25.3 |
| expected tokens | {"total_input_tokens": 22224, "total_output_tokens": 102108, "total_thought_tokens": 91920, "total_cached_tokens": 0, "total_tool_use_tokens": 0, "total_tokens": 216252} |

> Derived from ONE observed batch. The range is a planning band, not a prediction, and a single observation supports no confidence interval.

> Free Tier: HIGHER PER REQUEST - each batch produces ~32x the output of a singleton call, so a per-minute TOKEN limit, not a request limit, becomes the plausible binding constraint QUOTA_SNAPSHOT_NOT_PROGRAMMATICALLY_AVAILABLE - the Free Tier RPM/TPM/RPD limits for this project must be read from AI Studio before C3. No number was invented here.

## Recommendation

**C3 must not be frozen on this evidence** - but read the failure precisely, because
it is not where the milestone's name suggests.

What worked, measured rather than asserted:

- one request returned exactly 32 objects in a single structured response;
- all 32 parsed, satisfied the strict item schema, the ontology, every range and
  every compatibility rule, and were accepted with zero duplicates;
- the archived response replays offline to identical identities;
- the generic quotas were met on every axis, required and preferred, and the
  singleton mode collapse is gone.

What failed is a single criterion, `no_compiler_failures`, and its cause is the
validator/compiler disagreement over `generator_route` described above - not the
batch shape, not the envelope, and not the quotas. The batch mechanism itself is
sound; the recipe-acceptance contract has a gap that C3 would inherit.

Failed criteria: ['no_compiler_failures']

**No freeze was performed. C3 was not started.**

