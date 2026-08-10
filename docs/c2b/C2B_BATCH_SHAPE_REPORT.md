# C2B - 32-recipe batch-shape validation

One logical Gemini request asking for 32 recipe objects in a single structured
response, under generic ontology-level coverage quotas. This is a disposable
development experiment: nothing here enters C3, the final 256-recipe bank, a
synthetic bank or detector training.

**Outcome: BATCH_SHAPE_FAIL**

## Batch contract

| field | value |
| --- | --- |
| provider | Google Gemini Developer API |
| model | `gemini-3.6-flash` |
| API surface | interactions |
| thinking_level | medium |
| system prompt identity | `d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f` |
| system prompt changed in C2B | False |
| batch generation-template identity | `e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20` |
| coverage quota identity | `89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5` |
| single-recipe (item) schema identity | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` |
| batch-envelope schema identity (sent) | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` |
| ontology identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| provider config identity | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` |
| allow_ontology_aliases | False |
| recipes per request | 32 |

### Two schema identities, and why they differ from C1's record

> The value recorded at C1 as `llm_schema_identity_12x32` (7afc3abd...) is the 32-OBJECT ENVELOPE identity, not a single-recipe identity. The single-recipe ITEM schema is a separate value (1e3f050e...) and is byte-identical in the 1-object envelope C2 used and the envelope C2B sends, which is the evidence that changing the batch size did not touch recipe semantics.

> **Array-bounds finding.** The C1-recorded 32-object envelope carries minItems=maxItems=32 and the provider REJECTS it with 400 INVALID_ARGUMENT. The byte-identical envelope with the bound at 1 was accepted 42 times during C2, and the two schemas differ by nothing else (2695 vs 2697 bytes, identical item schema). Google documents array length limits as a source of schema-complexity rejection. C2B therefore omits the bound from the REQUEST only; `validate_response` still rejects any response whose recipe count is not exactly 32, so the scientific requirement is unchanged and is enforced where it always was.

| schema | identity | status |
| --- | --- | --- |
| single-recipe item schema | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` | unchanged from C2 - byte-identical in both envelopes |
| C2 singleton envelope (n=1) | `e9f66067c2de2deda5373a99dc6c92689c0ab2d2163b80adcde57af83df9bbd1` | what C2 sent, accepted 42 times |
| C1-recorded 32-object envelope | `7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4` | **rejected by the provider, 400 INVALID_ARGUMENT** |
| C2B batch envelope (sent) | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` | same schema without the array length bound |

## Live batch

| measure | value |
| --- | --- |
| logical batches executed | 1 |
| second batch issued | False |
| provider attempts | 1 |
| transport retries | 0 |
| 429 events | 0 |
| status | COMPLETE |
| requested objects | 32 |
| returned objects | 32 |

| provenance field | value |
| --- | --- |
| latency (s) | 63.218704 |
| model revision | gemini-3.6-flash |
| finish reason | completed |
| raw response sha256 | `46d4aad224b68548b8ad0f77043b3a63286c7da47f5003e6be438dcddd34b7df` |
| request sha256 | `13957aff371ae4f51a1c259167e6729a4adccfe5823d09d62cd35b1a1e741d11` |
| total_cached_tokens | 0 |
| total_input_tokens | 1852 |
| total_output_tokens | 8509 |
| total_thought_tokens | 7660 |
| total_tokens | 18021 |
| total_tool_use_tokens | 0 |

## Batch structure

| measure | value |
| --- | --- |
| requested objects | 32 |
| returned objects | 32 |
| valid (accepted) | 32 |
| invalid | 0 |
| duplicates | 0 |
| compiler failures | 10 |
| response-level issues | 0 |

- offline replay identical to the live run: **True**
- distinct structural patterns: 32/32

## Generator route: valid recipes the compiler cannot build

| generator_route | recipes |
| --- | --- |
| gpat | 10 |
| physics | 16 |
| physics+gpat | 6 |

- accepted objects without the physics route: **10/32** (batch indices [2, 5, 8, 11, 13, 16, 19, 22, 26, 29])
- compiler failures explained entirely by that: **True**
- caused by the coverage quotas: **False** - The C2B quotas constrain media, geometry, illumination, artifacts and regions. They never mention generator_route, so they cannot have forced this choice.

Every one of the 32 C2 singleton recipes declared the physics route (25 physics+gpat, 7 physics-only). The batch request is the first context in which the model chose gpat-only.

**Consequence for C3.** A gpat-only recipe passes validation and then cannot be compiled into an operator graph, so it cannot reach the physics renderer. C3 must decide explicitly whether the physics route is required, or whether a gpat-only recipe is a separate accepted class. This is a USER decision; C2B changes nothing.

## Coverage and quota compliance

### artifacts - 8/8 present, max share 16.4179% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| halftone | 7 | 21.875 | 1 | 4 | 16 | pass |
| pixel_grid | 4 | 12.5 | 1 | 4 | 16 | pass |
| moire | 7 | 21.875 | 1 | 4 | 16 | pass |
| specular_reflection | 11 | 34.375 | 1 | 4 | 16 | pass |
| texture_smoothing | 10 | 31.25 | 1 | 4 | 16 | pass |
| color_shift | 9 | 28.125 | 1 | 4 | 16 | pass |
| boundary_inconsistency | 9 | 28.125 | 1 | 4 | 16 | pass |
| blur | 10 | 31.25 | 1 | 4 | 16 | pass |

### geometry - 6/6 present, max share 18.75% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| flat | 6 | 18.75 | 3 | - | 8 | pass |
| curved | 5 | 15.625 | 3 | - | 8 | pass |
| partial-curved | 5 | 15.625 | 3 | - | 8 | pass |
| flexible | 5 | 15.625 | 3 | - | 8 | pass |
| rigid | 5 | 15.625 | 3 | - | 8 | pass |
| boundary-only | 6 | 18.75 | 3 | - | 8 | pass |

### illumination - 6/6 present, max share 18.75% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| front | 6 | 18.75 | 3 | - | 8 | pass |
| left | 6 | 18.75 | 3 | - | 8 | pass |
| right | 5 | 15.625 | 3 | - | 8 | pass |
| top | 5 | 15.625 | 3 | - | 8 | pass |
| bottom | 5 | 15.625 | 3 | - | 8 | pass |
| mixed | 5 | 15.625 | 3 | - | 8 | pass |

### media - 5/5 present, max share 21.875% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| paper-like | 6 | 18.75 | 4 | - | 10 | pass |
| display-like | 6 | 18.75 | 4 | - | 10 | pass |
| plastic-like | 6 | 18.75 | 4 | - | 10 | pass |
| fabric-like | 7 | 21.875 | 4 | - | 10 | pass |
| reflective-film-like | 7 | 21.875 | 4 | - | 10 | pass |

### regions - 9/9 present, max share 12.1622% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| left_eye | 9 | 28.125 | 1 | 3 | 16 | pass |
| right_eye | 9 | 28.125 | 1 | 3 | 16 | pass |
| nose | 9 | 28.125 | 1 | 3 | 16 | pass |
| mouth | 8 | 25.0 | 1 | 3 | 16 | pass |
| forehead | 8 | 25.0 | 1 | 3 | 16 | pass |
| left_cheek | 7 | 21.875 | 1 | 3 | 16 | pass |
| right_cheek | 8 | 25.0 | 1 | 3 | 16 | pass |
| face_boundary | 9 | 28.125 | 1 | 3 | 16 | pass |
| context | 7 | 21.875 | 1 | 3 | 16 | pass |

| measure | min | max | mean | histogram |
| --- | --- | --- | --- | --- |
| artifacts per recipe | 2 | 3 | 2.09375 | {"2": 29, "3": 3} |
| regions per recipe | 1 | 3 | 2.3125 | {"1": 3, "2": 16, "3": 13} |

## C2 singleton versus C2B batch

| axis | categories | C2 present | C2B present | C2 max share | C2B max share | collapse resolved |
| --- | --- | --- | --- | --- | --- | --- |
| artifacts | 8 | 6 | 8 | 30.2326% | 16.4179% | yes |
| regions | 9 | 8 | 9 | 32.6316% | 12.1622% | yes |
| media | 5 | 2 | 5 | 81.25% | 21.875% | yes |
| geometry | 6 | 2 | 6 | 84.375% | 18.75% | yes |
| illumination | 6 | 2 | 6 | 81.25% | 18.75% | yes |

**Verdict: resolved on every axis.** 5/5 axes are now fully covered.

C2 singleton pilot versus the C2B batch, both source-independent prompt-development evidence. No dataset, target metric or attack taxonomy was consulted.

## Co-occurrence

### artifacts_x_media

| artifacts | paper-like | display-like | plastic-like | fabric-like | reflective-film-like | total |
| --- | --- | --- | --- | --- | --- | --- |
| halftone | 3 | 0 | 0 | 4 | 0 | 7 |
| pixel_grid | 0 | 4 | 0 | 0 | 0 | 4 |
| moire | 0 | 3 | 0 | 0 | 4 | 7 |
| specular_reflection | 0 | 2 | 4 | 0 | 5 | 11 |
| texture_smoothing | 2 | 2 | 2 | 3 | 1 | 10 |
| color_shift | 2 | 1 | 2 | 2 | 2 | 9 |
| boundary_inconsistency | 2 | 1 | 2 | 2 | 2 | 9 |
| blur | 3 | 1 | 2 | 3 | 1 | 10 |

occupied cells: 28/40 · dominant: specular_reflection x reflective-film-like (5, 7.4627%)

### artifacts_x_geometry

| artifacts | flat | curved | partial-curved | flexible | rigid | boundary-only | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 2 | 0 | 2 | 1 | 2 | 0 | 7 |
| pixel_grid | 2 | 1 | 0 | 1 | 0 | 0 | 4 |
| moire | 3 | 0 | 2 | 0 | 1 | 1 | 7 |
| specular_reflection | 1 | 3 | 0 | 1 | 2 | 4 | 11 |
| texture_smoothing | 2 | 3 | 2 | 1 | 1 | 1 | 10 |
| color_shift | 2 | 1 | 2 | 2 | 1 | 1 | 9 |
| boundary_inconsistency | 1 | 1 | 1 | 1 | 2 | 3 | 9 |
| blur | 0 | 1 | 2 | 3 | 1 | 3 | 10 |

occupied cells: 39/48 · dominant: specular_reflection x boundary-only (4, 5.9701%)

### artifacts_x_regions

| artifacts | left_eye | right_eye | nose | mouth | forehead | left_cheek | right_cheek | face_boundary | context | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 1 | 1 | 16 |
| pixel_grid | 2 | 1 | 1 | 0 | 1 | 2 | 2 | 1 | 1 | 11 |
| moire | 3 | 2 | 0 | 2 | 3 | 1 | 2 | 3 | 2 | 18 |
| specular_reflection | 2 | 2 | 5 | 1 | 2 | 1 | 4 | 5 | 2 | 24 |
| texture_smoothing | 6 | 4 | 3 | 4 | 2 | 1 | 1 | 0 | 3 | 24 |
| color_shift | 2 | 3 | 2 | 5 | 3 | 2 | 0 | 2 | 1 | 20 |
| boundary_inconsistency | 1 | 1 | 3 | 2 | 3 | 1 | 4 | 4 | 3 | 22 |
| blur | 1 | 3 | 2 | 2 | 1 | 4 | 1 | 3 | 3 | 20 |

occupied cells: 68/72 · dominant: texture_smoothing x left_eye (6, 3.871%)

### media_x_geometry

| media | flat | curved | partial-curved | flexible | rigid | boundary-only | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| display-like | 2 | 1 | 1 | 1 | 1 | 0 | 6 |
| plastic-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| fabric-like | 1 | 1 | 1 | 2 | 1 | 1 | 7 |
| reflective-film-like | 1 | 1 | 1 | 0 | 1 | 3 | 7 |

occupied cells: 28/30 · dominant: reflective-film-like x boundary-only (3, 9.375%)

### media_x_illumination

| media | front | left | right | top | bottom | mixed | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| display-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| plastic-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| fabric-like | 2 | 1 | 1 | 1 | 1 | 1 | 7 |
| reflective-film-like | 1 | 2 | 1 | 1 | 1 | 1 | 7 |

occupied cells: 30/30 · dominant: reflective-film-like x left (2, 6.25%)

### geometry_x_illumination

| geometry | front | left | right | top | bottom | mixed | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat | 2 | 1 | 1 | 1 | 1 | 0 | 6 |
| curved | 0 | 1 | 1 | 1 | 1 | 1 | 5 |
| partial-curved | 1 | 0 | 1 | 1 | 1 | 1 | 5 |
| flexible | 1 | 1 | 0 | 1 | 1 | 1 | 5 |
| rigid | 1 | 1 | 1 | 0 | 1 | 1 | 5 |
| boundary-only | 1 | 2 | 1 | 1 | 0 | 1 | 6 |

occupied cells: 30/36 · dominant: flat x front (2, 6.25%)

## Physical validity classification

| classification | count |
| --- | --- |
| VALID_AND_QUOTA_COMPLIANT | 32 |
| VALID_BUT_QUOTA_MISS | 0 |
| INVALID | 0 |

No recipe was moved between categories after generation and no semantic field was
edited to satisfy a quota. Compatibility outranks quota by construction.

## All 32 returned objects

### index 0

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000000 |
| artifact(s) | halftone, texture_smoothing |
| strengths | halftone=0.25, texture_smoothing=0.2 (total 0.45) |
| region(s) | left_eye, right_eye |
| medium | paper-like (transparency 0.0, roughness 0.65) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| canonical identity | `c85848da0cc7378187998a09da4bbdcafbdca9916adbccdc3dad0b6f75755cdc` |
| graph hash | `e539e20679319e279be0c363fa4dd39477150af8fea311fcdcfe250e983704e7` |

*Physical reading:* a printed paper surface (held flat, transparency 0.0, roughness 0.65) covering left_eye, right_eye at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, under frontal light at yaw 0.0 deg, scale 1.0, compression q85.

### index 1

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000001 |
| artifact(s) | color_shift, blur |
| strengths | color_shift=0.15, blur=0.2 (total 0.35) |
| region(s) | nose, mouth |
| medium | paper-like (transparency 0.05, roughness 0.55) |
| geometry | curved (rigidity 0.4, coverage 0.75) |
| illumination | left |
| canonical identity | `65edc35084163c91c35c0ff8c7ba00969aaee6989c1b9330781efe0ff3d3a521` |
| graph hash | `55e76487e6a020181496b952dac9366862c845c12c75aa2a9715524d2174f4fe` |

*Physical reading:* a printed paper surface (curved, transparency 0.05, roughness 0.55) covering nose, mouth at coverage 0.75, producing color_shift at 0.15, blur at 0.2, under light from the left at yaw -15.0 deg, scale 0.95, compression q75.

### index 2

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000002 |
| artifact(s) | boundary_inconsistency, halftone |
| strengths | boundary_inconsistency=0.3, halftone=0.15 (total 0.45) |
| region(s) | forehead, left_cheek, right_cheek |
| medium | paper-like (transparency 0.02, roughness 0.7) |
| geometry | partial-curved (rigidity 0.5, coverage 0.6) |
| illumination | right |
| canonical identity | `1c7289695c497654fb9eafb370c84910db48f364570b17df3bf5cd72d50fb78b` |
| graph hash | `None` |

*Physical reading:* a printed paper surface (partly curved, transparency 0.02, roughness 0.7) covering forehead, left_cheek, right_cheek at coverage 0.6, producing boundary_inconsistency at 0.3, halftone at 0.15, under light from the right at yaw 20.0 deg, scale 1.05, compression q90.

### index 3

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000003 |
| artifact(s) | texture_smoothing, color_shift |
| strengths | texture_smoothing=0.35, color_shift=0.1 (total 0.45) |
| region(s) | left_eye, nose, mouth |
| medium | paper-like (transparency 0.0, roughness 0.8) |
| geometry | flexible (rigidity 0.2, coverage 0.9) |
| illumination | top |
| canonical identity | `b371e6fd1e3d3e2990d3d3d4e2805291936ec14e270a2759d980d58f18829644` |
| graph hash | `8f2190ea55a6098cf77be2a201d80e2685580f4c57b85c4a6c86eab73ac037a2` |

*Physical reading:* a printed paper surface (flexible and deformable, transparency 0.0, roughness 0.8) covering left_eye, nose, mouth at coverage 0.9, producing texture_smoothing at 0.35, color_shift at 0.1, under light from above at yaw 5.0 deg, scale 0.9, compression q65.

### index 4

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000004 |
| artifact(s) | halftone, blur |
| strengths | halftone=0.4, blur=0.15 (total 0.55) |
| region(s) | right_eye, forehead |
| medium | paper-like (transparency 0.01, roughness 0.5) |
| geometry | rigid (rigidity 0.95, coverage 0.7) |
| illumination | bottom |
| canonical identity | `4a9a55634c203c4031d87967124f1012a4fa597f49f4a8c20fe8c9c2a20a5868` |
| graph hash | `aa626602f26effff90881e4b1697a745756fa3f6b5a8c64f43f616692d7b8bf4` |

*Physical reading:* a printed paper surface (rigid, transparency 0.01, roughness 0.5) covering right_eye, forehead at coverage 0.7, producing halftone at 0.4, blur at 0.15, under light from below at yaw -10.0 deg, scale 1.1, compression q80.

### index 5

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000005 |
| artifact(s) | boundary_inconsistency, blur |
| strengths | boundary_inconsistency=0.25, blur=0.2 (total 0.45) |
| region(s) | face_boundary, context |
| medium | paper-like (transparency 0.08, roughness 0.6) |
| geometry | boundary-only (rigidity 0.3, coverage 0.3) |
| illumination | mixed |
| canonical identity | `70d4de324927f00ee5c4e324b7dd7b24565a456664b451f2d1e310103b969f65` |
| graph hash | `None` |

*Physical reading:* a printed paper surface (present only at the face edge, transparency 0.08, roughness 0.6) covering face_boundary, context at coverage 0.3, producing boundary_inconsistency at 0.25, blur at 0.2, under mixed lighting at yaw 12.0 deg, scale 0.85, compression q70.

### index 6

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000006 |
| artifact(s) | pixel_grid, moire |
| strengths | pixel_grid=0.3, moire=0.2 (total 0.5) |
| region(s) | left_cheek, right_cheek, face_boundary |
| medium | display-like (transparency 0.0, roughness 0.02) |
| geometry | flat (rigidity 0.9, coverage 0.95) |
| illumination | left |
| canonical identity | `448f84cc484b026ebe6e71363fb0a9f31934c3b0275820b7c4798d23b598cdf3` |
| graph hash | `9806d79b15b3eb19183a4377b48460582d722bef19914e04b7de8597f6d648e5` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.02) covering left_cheek, right_cheek, face_boundary at coverage 0.95, producing pixel_grid at 0.3, moire at 0.2, under light from the left at yaw -25.0 deg, scale 1.0, compression q95.

### index 7

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000007 |
| artifact(s) | specular_reflection, pixel_grid |
| strengths | specular_reflection=0.25, pixel_grid=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.01, roughness 0.05) |
| geometry | curved (rigidity 0.7, coverage 0.8) |
| illumination | right |
| canonical identity | `ca754c377697bbd1e34d09d11b89c4860da099f57684656a5f6bd6b190e0c86c` |
| graph hash | `b0ae9767b0f8167daa5672e23c7f3e6c3a4e2c80032210bb1b6e7a54dc31ed04` |

*Physical reading:* an emissive display panel (curved, transparency 0.01, roughness 0.05) covering left_eye, right_eye, nose at coverage 0.8, producing specular_reflection at 0.25, pixel_grid at 0.2, under light from the right at yaw 18.0 deg, scale 1.15, compression q88.

### index 8

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000008 |
| artifact(s) | moire, texture_smoothing, color_shift |
| strengths | moire=0.25, texture_smoothing=0.15, color_shift=0.1 (total 0.5) |
| region(s) | mouth, forehead |
| medium | display-like (transparency 0.0, roughness 0.01) |
| geometry | partial-curved (rigidity 0.6, coverage 0.85) |
| illumination | top |
| canonical identity | `7f5a6d569ca2ec5b02ddadd2359e223b44b0a582bc6a0171c6a2f85c939fe369` |
| graph hash | `None` |

*Physical reading:* an emissive display panel (partly curved, transparency 0.0, roughness 0.01) covering mouth, forehead at coverage 0.85, producing moire at 0.25, texture_smoothing at 0.15, color_shift at 0.1, under light from above at yaw 0.0 deg, scale 0.8, compression q60.

### index 9

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000009 |
| artifact(s) | pixel_grid, blur |
| strengths | pixel_grid=0.35, blur=0.15 (total 0.5) |
| region(s) | left_cheek, right_cheek |
| medium | display-like (transparency 0.02, roughness 0.08) |
| geometry | flexible (rigidity 0.15, coverage 0.65) |
| illumination | bottom |
| canonical identity | `a6f3a8d17f0c529705718c576447c4a5e868ec51c163ec77a7c3282714cb2897` |
| graph hash | `14bcb00b49ec9c427173176b8cc55d3e00dc95682c93aeb5c0ee57f90f23ead6` |

*Physical reading:* an emissive display panel (flexible and deformable, transparency 0.02, roughness 0.08) covering left_cheek, right_cheek at coverage 0.65, producing pixel_grid at 0.35, blur at 0.15, under light from below at yaw -30.0 deg, scale 1.2, compression q50.

### index 10

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000010 |
| artifact(s) | specular_reflection, boundary_inconsistency |
| strengths | specular_reflection=0.3, boundary_inconsistency=0.2 (total 0.5) |
| region(s) | nose, mouth, forehead |
| medium | display-like (transparency 0.0, roughness 0.03) |
| geometry | rigid (rigidity 0.85, coverage 0.9) |
| illumination | mixed |
| canonical identity | `ac3949baea1d0eae5527b41ac95c1ac2bc6c58c046fc78d87f0bd6549d87278d` |
| graph hash | `bf57cba43a57314997c1712488929289821764b99f28e427aae5a34d100e4e65` |

*Physical reading:* an emissive display panel (rigid, transparency 0.0, roughness 0.03) covering nose, mouth, forehead at coverage 0.9, producing specular_reflection at 0.3, boundary_inconsistency at 0.2, under mixed lighting at yaw 8.0 deg, scale 1.0, compression q82.

### index 11

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000011 |
| artifact(s) | pixel_grid, moire, texture_smoothing |
| strengths | pixel_grid=0.25, moire=0.15, texture_smoothing=0.2 (total 0.6) |
| region(s) | left_eye, forehead, context |
| medium | display-like (transparency 0.0, roughness 0.01) |
| geometry | flat (rigidity 0.95, coverage 1.0) |
| illumination | front |
| canonical identity | `27f72c171ec5f210a8b7722094bfcb321e6be8271ba5422ebf485f2db3738ed4` |
| graph hash | `None` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.01) covering left_eye, forehead, context at coverage 1.0, producing pixel_grid at 0.25, moire at 0.15, texture_smoothing at 0.2, under frontal light at yaw 0.0 deg, scale 1.0, compression q100.

### index 12

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000012 |
| artifact(s) | specular_reflection, texture_smoothing |
| strengths | specular_reflection=0.4, texture_smoothing=0.2 (total 0.6) |
| region(s) | right_eye, nose |
| medium | plastic-like (transparency 0.3, roughness 0.1) |
| geometry | curved (rigidity 0.65, coverage 0.8) |
| illumination | top |
| canonical identity | `c6d536d357a6a3d951957c9cd22679e39f74344df810421ae2cbc3904b9b833c` |
| graph hash | `b82af48a83d18323099cd370ea7620ab3f285f55a7bdb46d2d6f7d0e02562fd4` |

*Physical reading:* a moulded plastic surface (curved, transparency 0.3, roughness 0.1) covering right_eye, nose at coverage 0.8, producing specular_reflection at 0.4, texture_smoothing at 0.2, under light from above at yaw 15.0 deg, scale 1.05, compression q78.

### index 13

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000013 |
| artifact(s) | color_shift, blur |
| strengths | color_shift=0.2, blur=0.25 (total 0.45) |
| region(s) | mouth, left_cheek |
| medium | plastic-like (transparency 0.5, roughness 0.15) |
| geometry | partial-curved (rigidity 0.45, coverage 0.7) |
| illumination | bottom |
| canonical identity | `accb38361c806b75dc6896745faeeba133954a6b5c45fc7bab1890867725fafe` |
| graph hash | `None` |

*Physical reading:* a moulded plastic surface (partly curved, transparency 0.5, roughness 0.15) covering mouth, left_cheek at coverage 0.7, producing color_shift at 0.2, blur at 0.25, under light from below at yaw -20.0 deg, scale 0.95, compression q65.

### index 14

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000014 |
| artifact(s) | specular_reflection, boundary_inconsistency |
| strengths | specular_reflection=0.3, boundary_inconsistency=0.15 (total 0.45) |
| region(s) | right_cheek, face_boundary |
| medium | plastic-like (transparency 0.2, roughness 0.2) |
| geometry | flexible (rigidity 0.1, coverage 0.85) |
| illumination | mixed |
| canonical identity | `6bbfe02661f96379de18d41f09f30fd5946bbb0f72ccd897af058401eed32ec7` |
| graph hash | `b306239e135366dad36f3ab554e808e49a09af4d4b95e6dd2f59b977e2b2b985` |

*Physical reading:* a moulded plastic surface (flexible and deformable, transparency 0.2, roughness 0.2) covering right_cheek, face_boundary at coverage 0.85, producing specular_reflection at 0.3, boundary_inconsistency at 0.15, under mixed lighting at yaw 35.0 deg, scale 1.1, compression q85.

### index 15

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000015 |
| artifact(s) | texture_smoothing, color_shift |
| strengths | texture_smoothing=0.3, color_shift=0.15 (total 0.45) |
| region(s) | left_eye, right_eye, context |
| medium | plastic-like (transparency 0.1, roughness 0.05) |
| geometry | rigid (rigidity 0.9, coverage 0.75) |
| illumination | front |
| canonical identity | `d5b9275df763c1a980f57fb615781859d8b4d4f01f26b2c077c50d3cf756ce1a` |
| graph hash | `0b3815b4c36e9cc05cfff70af749888c2078fcfc19a572b96b44ed54cabd0089` |

*Physical reading:* a moulded plastic surface (rigid, transparency 0.1, roughness 0.05) covering left_eye, right_eye, context at coverage 0.75, producing texture_smoothing at 0.3, color_shift at 0.15, under frontal light at yaw 0.0 deg, scale 1.0, compression q92.

### index 16

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000016 |
| artifact(s) | specular_reflection, blur |
| strengths | specular_reflection=0.35, blur=0.2 (total 0.55) |
| region(s) | face_boundary |
| medium | plastic-like (transparency 0.6, roughness 0.25) |
| geometry | boundary-only (rigidity 0.35, coverage 0.25) |
| illumination | left |
| canonical identity | `f9724b1b5fdd2e7923b1a9f2da71e001a57cc01c7d55d0cb1acb9dcdaf521f8f` |
| graph hash | `None` |

*Physical reading:* a moulded plastic surface (present only at the face edge, transparency 0.6, roughness 0.25) covering face_boundary at coverage 0.25, producing specular_reflection at 0.35, blur at 0.2, under light from the left at yaw -10.0 deg, scale 0.88, compression q72.

### index 17

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000017 |
| artifact(s) | boundary_inconsistency, specular_reflection |
| strengths | boundary_inconsistency=0.25, specular_reflection=0.2 (total 0.45) |
| region(s) | nose, forehead, right_cheek |
| medium | plastic-like (transparency 0.15, roughness 0.08) |
| geometry | flat (rigidity 0.8, coverage 0.95) |
| illumination | right |
| canonical identity | `920ba9ac531c2f1270c3d26807f80b76a6873721ea1f65bebdbb9f99c433175d` |
| graph hash | `15d530afc11e4bc051e0b0f608f06662717de00e0595a2966cc994f35401bf78` |

*Physical reading:* a moulded plastic surface (held flat, transparency 0.15, roughness 0.08) covering nose, forehead, right_cheek at coverage 0.95, producing boundary_inconsistency at 0.25, specular_reflection at 0.2, under light from the right at yaw 22.0 deg, scale 1.02, compression q80.

### index 18

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000018 |
| artifact(s) | halftone, texture_smoothing |
| strengths | halftone=0.3, texture_smoothing=0.25 (total 0.55) |
| region(s) | left_eye, mouth |
| medium | fabric-like (transparency 0.0, roughness 0.85) |
| geometry | partial-curved (rigidity 0.3, coverage 0.8) |
| illumination | front |
| canonical identity | `ae0428be6e231b80fbd7b98efcc7c57e23e097e771c971cc2996294179d0c7c9` |
| graph hash | `dbbfa45b103c24dc72d9d32d4ead65e826813d7d3233826b0481913b44c624ae` |

*Physical reading:* a woven fabric surface (partly curved, transparency 0.0, roughness 0.85) covering left_eye, mouth at coverage 0.8, producing halftone at 0.3, texture_smoothing at 0.25, under frontal light at yaw 0.0 deg, scale 1.0, compression q70.

### index 19

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000019 |
| artifact(s) | color_shift, blur |
| strengths | color_shift=0.25, blur=0.2 (total 0.45) |
| region(s) | right_eye, left_cheek |
| medium | fabric-like (transparency 0.1, roughness 0.75) |
| geometry | flexible (rigidity 0.05, coverage 0.85) |
| illumination | left |
| canonical identity | `2476fdadf1f7c757452f92210912691715f85f2b933c124a182a336b772d0a7c` |
| graph hash | `None` |

*Physical reading:* a woven fabric surface (flexible and deformable, transparency 0.1, roughness 0.75) covering right_eye, left_cheek at coverage 0.85, producing color_shift at 0.25, blur at 0.2, under light from the left at yaw -12.0 deg, scale 0.92, compression q60.

### index 20

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000020 |
| artifact(s) | halftone, boundary_inconsistency |
| strengths | halftone=0.2, boundary_inconsistency=0.25 (total 0.45) |
| region(s) | nose, right_cheek |
| medium | fabric-like (transparency 0.02, roughness 0.9) |
| geometry | rigid (rigidity 0.7, coverage 0.65) |
| illumination | right |
| canonical identity | `87082890487a09567dc5c9bf5a03e2e0e4597b80108e11d039dc2241c09d4cc0` |
| graph hash | `4c6b4a16c5e71bbd02afdb7cbe3b5a9a0d23a00a409a773932e778acfa0022fc` |

*Physical reading:* a woven fabric surface (rigid, transparency 0.02, roughness 0.9) covering nose, right_cheek at coverage 0.65, producing halftone at 0.2, boundary_inconsistency at 0.25, under light from the right at yaw 28.0 deg, scale 1.08, compression q82.

### index 21

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000021 |
| artifact(s) | texture_smoothing, blur |
| strengths | texture_smoothing=0.3, blur=0.15 (total 0.45) |
| region(s) | context |
| medium | fabric-like (transparency 0.15, roughness 0.8) |
| geometry | boundary-only (rigidity 0.2, coverage 0.2) |
| illumination | top |
| canonical identity | `6eb36543bd71af4163b1a23cc12ec5f2ade314cd571a5bd997b02086cd09a631` |
| graph hash | `d07b5d970b980a88492ab523759e0a28897f97f4f551a73845974935ad8cfcf0` |

*Physical reading:* a woven fabric surface (present only at the face edge, transparency 0.15, roughness 0.8) covering context at coverage 0.2, producing texture_smoothing at 0.3, blur at 0.15, under light from above at yaw -5.0 deg, scale 0.82, compression q75.

### index 22

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000022 |
| artifact(s) | halftone, color_shift |
| strengths | halftone=0.35, color_shift=0.15 (total 0.5) |
| region(s) | forehead, face_boundary |
| medium | fabric-like (transparency 0.0, roughness 0.7) |
| geometry | flat (rigidity 0.6, coverage 0.9) |
| illumination | bottom |
| canonical identity | `582185c70bf595c7f862b821a0257c736cb7f41f20fe6f7fa113c9d5eace30ac` |
| graph hash | `None` |

*Physical reading:* a woven fabric surface (held flat, transparency 0.0, roughness 0.7) covering forehead, face_boundary at coverage 0.9, producing halftone at 0.35, color_shift at 0.15, under light from below at yaw 10.0 deg, scale 1.0, compression q88.

### index 23

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000023 |
| artifact(s) | boundary_inconsistency, texture_smoothing |
| strengths | boundary_inconsistency=0.3, texture_smoothing=0.2 (total 0.5) |
| region(s) | left_eye, right_eye, mouth |
| medium | fabric-like (transparency 0.05, roughness 0.65) |
| geometry | curved (rigidity 0.35, coverage 0.75) |
| illumination | mixed |
| canonical identity | `5e2494b84b2cefed1cbe2597281c353710e789633a621583b9f594acf010ac63` |
| graph hash | `13c7193f7e07df37650c2ad9340a3f9feecb9a278599a98e1ae7aadaac239afe` |

*Physical reading:* a woven fabric surface (curved, transparency 0.05, roughness 0.65) covering left_eye, right_eye, mouth at coverage 0.75, producing boundary_inconsistency at 0.3, texture_smoothing at 0.2, under mixed lighting at yaw -18.0 deg, scale 0.98, compression q68.

### index 24

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000024 |
| artifact(s) | halftone, blur |
| strengths | halftone=0.25, blur=0.25 (total 0.5) |
| region(s) | nose, left_cheek, context |
| medium | fabric-like (transparency 0.08, roughness 0.82) |
| geometry | flexible (rigidity 0.1, coverage 0.8) |
| illumination | front |
| canonical identity | `f690eec602cb2fdae5b5141a296e6b2a282eb5bfc468ae258bc7645c6925f51c` |
| graph hash | `9e5a4d9bd39f0bb21ea4533bf92cc9730786d671cf91ad5ec44cf31da271e05c` |

*Physical reading:* a woven fabric surface (flexible and deformable, transparency 0.08, roughness 0.82) covering nose, left_cheek, context at coverage 0.8, producing halftone at 0.25, blur at 0.25, under frontal light at yaw 2.0 deg, scale 1.04, compression q76.

### index 25

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000025 |
| artifact(s) | specular_reflection, moire |
| strengths | specular_reflection=0.4, moire=0.2 (total 0.6) |
| region(s) | left_eye, right_cheek |
| medium | reflective-film-like (transparency 0.2, roughness 0.02) |
| geometry | rigid (rigidity 0.9, coverage 0.85) |
| illumination | left |
| canonical identity | `20c6a0fde517c2a568cbc0797e43824476fc21247278e6307549225995caf1b7` |
| graph hash | `119796eb9e88cf11af6408295d88455240a4e3eaa4ecd09d26efbd3ad50e218a` |

*Physical reading:* a reflective film overlay (rigid, transparency 0.2, roughness 0.02) covering left_eye, right_cheek at coverage 0.85, producing specular_reflection at 0.4, moire at 0.2, under light from the left at yaw -35.0 deg, scale 1.12, compression q90.

### index 26

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000026 |
| artifact(s) | specular_reflection, boundary_inconsistency |
| strengths | specular_reflection=0.35, boundary_inconsistency=0.25 (total 0.6) |
| region(s) | face_boundary, context |
| medium | reflective-film-like (transparency 0.35, roughness 0.05) |
| geometry | boundary-only (rigidity 0.4, coverage 0.35) |
| illumination | right |
| canonical identity | `878d6dea011649b7fcebb03ab82165d734f084812ad01add60aa94327343ac84` |
| graph hash | `None` |

*Physical reading:* a reflective film overlay (present only at the face edge, transparency 0.35, roughness 0.05) covering face_boundary, context at coverage 0.35, producing specular_reflection at 0.35, boundary_inconsistency at 0.25, under light from the right at yaw 14.0 deg, scale 0.86, compression q84.

### index 27

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000027 |
| artifact(s) | moire, color_shift |
| strengths | moire=0.3, color_shift=0.15 (total 0.45) |
| region(s) | right_eye, mouth, forehead |
| medium | reflective-film-like (transparency 0.1, roughness 0.01) |
| geometry | flat (rigidity 0.95, coverage 0.95) |
| illumination | top |
| canonical identity | `70dcab5f8f79f518ccead169a5bc70249ac107e7f386e9db29b5da872231e581` |
| graph hash | `a0d1e48dd318fce2bfc6d5d6fe57a874f81e62b9f79062fe989e01b91d3164c6` |

*Physical reading:* a reflective film overlay (held flat, transparency 0.1, roughness 0.01) covering right_eye, mouth, forehead at coverage 0.95, producing moire at 0.3, color_shift at 0.15, under light from above at yaw 0.0 deg, scale 1.0, compression q96.

### index 28

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000028 |
| artifact(s) | specular_reflection, texture_smoothing |
| strengths | specular_reflection=0.3, texture_smoothing=0.25 (total 0.55) |
| region(s) | nose, left_cheek, right_cheek |
| medium | reflective-film-like (transparency 0.4, roughness 0.08) |
| geometry | curved (rigidity 0.5, coverage 0.7) |
| illumination | bottom |
| canonical identity | `982688c4a5a9b0985c1c9fc9af5db62be23723c497776f6c9d5d203b72ca48d8` |
| graph hash | `4336ab2124ab3ff20f0810f10d71c913b9b63b2fb3ed8eedfad45858634e2255` |

*Physical reading:* a reflective film overlay (curved, transparency 0.4, roughness 0.08) covering nose, left_cheek, right_cheek at coverage 0.7, producing specular_reflection at 0.3, texture_smoothing at 0.25, under light from below at yaw -22.0 deg, scale 0.94, compression q74.

### index 29

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | failed |
| recipe id | R-000029 |
| artifact(s) | moire, blur |
| strengths | moire=0.25, blur=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, face_boundary |
| medium | reflective-film-like (transparency 0.15, roughness 0.03) |
| geometry | partial-curved (rigidity 0.6, coverage 0.8) |
| illumination | mixed |
| canonical identity | `70038648337deb9086ea0c0ac9bad95a95c685a935751271fd34cae2a14401de` |
| graph hash | `None` |

*Physical reading:* a reflective film overlay (partly curved, transparency 0.15, roughness 0.03) covering left_eye, right_eye, face_boundary at coverage 0.8, producing moire at 0.25, blur at 0.2, under mixed lighting at yaw 30.0 deg, scale 1.18, compression q86.

### index 30

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000030 |
| artifact(s) | specular_reflection, color_shift |
| strengths | specular_reflection=0.45, color_shift=0.1 (total 0.55) |
| region(s) | face_boundary |
| medium | reflective-film-like (transparency 0.25, roughness 0.04) |
| geometry | boundary-only (rigidity 0.3, coverage 0.18) |
| illumination | front |
| canonical identity | `9cbd29271b8ec0bbcd7d89b0d21d7535424116ace53b88c855e772cb7f1ef149` |
| graph hash | `17666ac323f5180ce6e6fd18336acbf64c42413cd3678a2bdf07ff6dd72f6b36` |

*Physical reading:* a reflective film overlay (present only at the face edge, transparency 0.25, roughness 0.04) covering face_boundary at coverage 0.18, producing specular_reflection at 0.45, color_shift at 0.1, under frontal light at yaw 0.0 deg, scale 1.0, compression q91.

### index 31

| field | value |
| --- | --- |
| status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000031 |
| artifact(s) | specular_reflection, moire, boundary_inconsistency |
| strengths | specular_reflection=0.25, moire=0.2, boundary_inconsistency=0.2 (total 0.65) |
| region(s) | face_boundary, context |
| medium | reflective-film-like (transparency 0.3, roughness 0.06) |
| geometry | boundary-only (rigidity 0.25, coverage 0.4) |
| illumination | left |
| canonical identity | `8ec667d6a78a1b7cff44a7cc767bd7e13a10393e2c28ce1e8d638ab2b94c710b` |
| graph hash | `cf20a8532bcb4801185022c78fa2dd4e855f7c5ca4c2f0675e6c92bbbbb26eb7` |

*Physical reading:* a reflective film overlay (present only at the face edge, transparency 0.3, roughness 0.06) covering face_boundary, context at coverage 0.4, producing specular_reflection at 0.25, moire at 0.2, boundary_inconsistency at 0.2, under light from the left at yaw -8.0 deg, scale 0.78, compression q62.

## Verdict detail

| check | result |
| --- | --- |
| returned_exactly_32 | pass |
| no_response_level_issues | pass |
| semantic_validity_at_least_threshold | pass |
| no_compiler_failures | **FAIL** |
| duplicate_rate_within_threshold | pass |
| all_axes_fully_represented | pass |
| no_severe_mode_collapse | pass |
| quotas_substantially_satisfied | pass |
| quotas_did_not_force_incompatibility | pass |

Thresholds were fixed before the batch returned: `{"returned_objects_exact": 32, "min_semantic_validity": 0.9, "max_response_issues": 0, "max_compiler_failures": 0, "max_duplicate_rate": 0.1, "max_axis_share_percent": 60.0, "max_required_quota_failures": 2, "axes_requiring_full_presence": ["media", "geometry", "illumination", "artifacts", "regions"]}`

