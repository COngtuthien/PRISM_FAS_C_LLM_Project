# C2 - LLM recipe pilot report

32 disposable pilot slots generated live against the frozen C1 provider contract.
These recipes are C2 development artifacts. They never enter the C3 384 candidate
slots, the final 256-recipe LLM bank, a synthetic bank or detector training.

## Frozen contract

| field | value |
| --- | --- |
| provider | Google Gemini Developer API |
| model | `gemini-3.6-flash` |
| API surface | interactions |
| thinking_level | medium |
| prompt identity | `d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f` |
| schema identity (12x32 reference) | `7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4` |
| schema identity (per pilot slot, 1 recipe) | `e9f66067c2de2deda5373a99dc6c92689c0ab2d2163b80adcde57af83df9bbd1` |
| ontology identity | `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| provider config identity | `3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da` |
| allow_ontology_aliases | False |
| sampling controls sent | none (temperature / top_p / top_k never sent) |
| tools / grounding / URL context / file search / code execution | none passed |
| input | str (text only) |

> The frozen 12x32 schema identity is the reference recorded at C1 for the C3 schedule. A pilot slot asks for one recipe, so its schema instance carries minItems/maxItems = 1 and therefore a different SHA-256. The schema BUILDER, the ontology it is built from and every enum, range and rule inside it are unchanged; only the batch size differs.

## Pilot totals

| measure | value |
| --- | --- |
| slots | 32 |
| successful slots | 32 |
| failed / exhausted slots | 0 |
| total provider calls | 42 |
| calls per slot (mean) | 1.3125 |
| first-attempt-valid rate | 1.0 |
| eventual-valid rate | 1.0 |
| invalid rate (per response) | 0.0 |
| retry rate (slots that retried) | 0.0 |
| retry exhaustion rate | 0.0 |
| duplicate rate (per response) | 0.0 |
| schema violations | 0 |
| ontology violations | 0 |
| range violations | 0 |
| compatibility violations | 0 |
| severity-budget violations | 0 |
| duplicate rejections | 0 |
| compiler failures | 0 |
| transport / API errors | 10 |
| transient 429 | 8 |
| quota exhausted | 2 |
| latency mean / median / p95 (s) | 7.742103 / 8.42513 / 11.021597 |

Token usage reported by the surface, summed over every attempt:

| counter | total |
| --- | --- |
| total_cached_tokens | 0 |
| total_input_tokens | 46432 |
| total_output_tokens | 10441 |
| total_thought_tokens | 50449 |
| total_tokens | 107322 |
| total_tool_use_tokens | 0 |

## Offline replay verification

- slots compared: 32
- replay identical to the live run: **True**
- mismatches: 0

The archived raw responses were re-parsed, re-validated, re-canonicalized, re-hashed
and re-compiled by the replay provider, which holds no client and no credential.

## Coverage

### artifacts (6/8 present, coverage 0.75)

| category | recipes | % of recipes | assignments | % of axis |
| --- | --- | --- | --- | --- |
| halftone | 6 | 18.75 | 6 | 6.9767 |
| pixel_grid | 26 | 81.25 | 26 | 30.2326 |
| moire | 25 | 78.125 | 25 | 29.0698 |
| specular_reflection | 18 | 56.25 | 18 | 20.9302 |
| texture_smoothing | 6 | 18.75 | 6 | 6.9767 |
| color_shift | 5 | 15.625 | 5 | 5.814 |
| boundary_inconsistency | 0 | 0.0 | 0 | 0.0 |
| blur | 0 | 0.0 | 0 | 0.0 |

missing: ['boundary_inconsistency', 'blur']

### regions (8/9 present, coverage 0.888889)

| category | recipes | % of recipes | assignments | % of axis |
| --- | --- | --- | --- | --- |
| left_eye | 31 | 96.875 | 31 | 32.6316 |
| right_eye | 19 | 59.375 | 19 | 20.0 |
| nose | 30 | 93.75 | 30 | 31.5789 |
| mouth | 5 | 15.625 | 5 | 5.2632 |
| forehead | 5 | 15.625 | 5 | 5.2632 |
| left_cheek | 1 | 3.125 | 1 | 1.0526 |
| right_cheek | 1 | 3.125 | 1 | 1.0526 |
| face_boundary | 3 | 9.375 | 3 | 3.1579 |
| context | 0 | 0.0 | 0 | 0.0 |

missing: ['context']

### media (2/5 present, coverage 0.4)

| category | recipes | % of recipes | assignments | % of axis |
| --- | --- | --- | --- | --- |
| paper-like | 6 | 18.75 | 6 | 18.75 |
| display-like | 26 | 81.25 | 26 | 81.25 |
| plastic-like | 0 | 0.0 | 0 | 0.0 |
| fabric-like | 0 | 0.0 | 0 | 0.0 |
| reflective-film-like | 0 | 0.0 | 0 | 0.0 |

missing: ['plastic-like', 'fabric-like', 'reflective-film-like']

### geometry (2/6 present, coverage 0.333333)

| category | recipes | % of recipes | assignments | % of axis |
| --- | --- | --- | --- | --- |
| flat | 27 | 84.375 | 27 | 84.375 |
| curved | 5 | 15.625 | 5 | 15.625 |
| partial-curved | 0 | 0.0 | 0 | 0.0 |
| flexible | 0 | 0.0 | 0 | 0.0 |
| rigid | 0 | 0.0 | 0 | 0.0 |
| boundary-only | 0 | 0.0 | 0 | 0.0 |

missing: ['partial-curved', 'flexible', 'rigid', 'boundary-only']

### illumination (2/6 present, coverage 0.333333)

| category | recipes | % of recipes | assignments | % of axis |
| --- | --- | --- | --- | --- |
| front | 26 | 81.25 | 26 | 81.25 |
| left | 0 | 0.0 | 0 | 0.0 |
| right | 0 | 0.0 | 0 | 0.0 |
| top | 0 | 0.0 | 0 | 0.0 |
| bottom | 0 | 0.0 | 0 | 0.0 |
| mixed | 6 | 18.75 | 6 | 18.75 |

missing: ['left', 'right', 'top', 'bottom']

### Artifacts and regions per recipe

| measure | min | max | mean | histogram |
| --- | --- | --- | --- | --- |
| artifacts per recipe | 2 | 3 | 2.6875 | {"2": 10, "3": 22} |
| regions per recipe | 2 | 3 | 2.96875 | {"2": 1, "3": 31} |

This is a prompt/contract coverage audit over the frozen ontology only. It is not
compared against any dataset attack family, and no target information was consulted.

## Co-occurrence

### artifact_x_medium

| artifact | paper-like | display-like | plastic-like | fabric-like | reflective-film-like | total |
| --- | --- | --- | --- | --- | --- | --- |
| halftone | 6 | 0 | 0 | 0 | 0 | 6 |
| pixel_grid | 0 | 26 | 0 | 0 | 0 | 26 |
| moire | 0 | 25 | 0 | 0 | 0 | 25 |
| specular_reflection | 0 | 18 | 0 | 0 | 0 | 18 |
| texture_smoothing | 6 | 0 | 0 | 0 | 0 | 6 |
| color_shift | 5 | 0 | 0 | 0 | 0 | 5 |
| boundary_inconsistency | 0 | 0 | 0 | 0 | 0 | 0 |
| blur | 0 | 0 | 0 | 0 | 0 | 0 |

compatible cells occupied: 6/28 (cells the ontology forbids are structurally empty, not a coverage gap)

### artifact_x_geometry

| artifact | flat | curved | partial-curved | flexible | rigid | boundary-only | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 5 | 1 | 0 | 0 | 0 | 0 | 6 |
| pixel_grid | 22 | 4 | 0 | 0 | 0 | 0 | 26 |
| moire | 21 | 4 | 0 | 0 | 0 | 0 | 25 |
| specular_reflection | 14 | 4 | 0 | 0 | 0 | 0 | 18 |
| texture_smoothing | 5 | 1 | 0 | 0 | 0 | 0 | 6 |
| color_shift | 4 | 1 | 0 | 0 | 0 | 0 | 5 |
| boundary_inconsistency | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| blur | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

cells occupied: 12/48

### artifact_x_region

| artifact | left_eye | right_eye | nose | mouth | forehead | left_cheek | right_cheek | face_boundary | context | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 5 | 1 | 5 | 2 | 3 | 1 | 1 | 0 | 0 | 18 |
| pixel_grid | 26 | 18 | 25 | 3 | 2 | 0 | 0 | 3 | 0 | 77 |
| moire | 25 | 18 | 24 | 3 | 1 | 0 | 0 | 3 | 0 | 74 |
| specular_reflection | 18 | 11 | 17 | 3 | 2 | 0 | 0 | 2 | 0 | 53 |
| texture_smoothing | 5 | 1 | 5 | 2 | 3 | 1 | 1 | 0 | 0 | 18 |
| color_shift | 5 | 1 | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 15 |
| boundary_inconsistency | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| blur | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

cells occupied: 37/72

## Flags

| flag | where | detail |
| --- | --- | --- |
| LOW_COVERAGE | media | 2/5 categories present; missing ['plastic-like', 'fabric-like', 'reflective-film-like'] |
| MODE_COLLAPSE | media | 'display-like' holds 81.25% of the media assignments |
| LOW_COVERAGE | geometry | 2/6 categories present; missing ['partial-curved', 'flexible', 'rigid', 'boundary-only'] |
| MODE_COLLAPSE | geometry | 'flat' holds 84.375% of the geometry assignments |
| LOW_COVERAGE | illumination | 2/6 categories present; missing ['left', 'right', 'top', 'bottom'] |
| MODE_COLLAPSE | illumination | 'front' holds 81.25% of the illumination assignments |
| REPEATED_PATTERN | - | {"artifacts": ["moire", "pixel_grid", "specular_reflection"], "geometry": "flat", "illumination": "front", "medium": "di |
| REPEATED_PATTERN | - | {"artifacts": ["moire", "pixel_grid"], "geometry": "flat", "illumination": "front", "medium": "display-like", "regions": |

Flags are objective and source-independent. No recipe is rated for usefulness against
any target.

## All 32 pilot slot outcomes

### pilot_000

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000000 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.2) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `2383e4bd5cc2181f5bbc22a5035da9cc144283080d008eef3064436a3d0631e6` |
| graph hash | `aa6ed1d58d050cce5a239c10cff08ffb7fcefb7ecce0d73f5f3796d7152a2d3d` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.2) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_001

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000001 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `4ccbc9b63bbe4dba0b5b6b0f2d1f0ed42fdaa5592f283ea25222ef9debeb6c40` |
| graph hash | `fd30553fe1a749fc216178306601f2da79c0fb3f1cae81f260542e03e533880f` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_002

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000002 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.35, moire=0.25 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.0, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.8) |
| illumination | front |
| recipe identity | `a0fe04d14f13d863e3fd4fa48e19d0597b95b446b063692d8470dd0fa920ca7e` |
| graph hash | `f840ad14b44ccf98c7b7077b59fe6df4d8d47f9e6452adc309db65cff39c876e` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.8, producing pixel_grid at 0.35, moire at 0.25, captured under frontal light at yaw 0.0 deg, scale 1.0, compression q85.

### pilot_003

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000003 |
| artifact type(s) | halftone, texture_smoothing |
| strength / severity | halftone=0.25, texture_smoothing=0.2 (total 0.45) |
| region(s) | forehead, left_cheek, right_cheek |
| medium | paper-like (transparency 0.0, roughness 0.6) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| recipe identity | `67e2b18c17d6772cdaef196419dd7bba757ac10c69afb65ded530cac3139f6cc` |
| graph hash | `c7d9cabafdf7ba7b64482a7b2a30cc0e8002df9f5ac242e2cb757e50b159d046` |

*Physical reading:* a printed paper surface (held flat, transparency 0.0, roughness 0.6) covering forehead, left_cheek, right_cheek at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_004

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000004 |
| artifact type(s) | moire, pixel_grid, specular_reflection |
| strength / severity | moire=0.25, pixel_grid=0.3, specular_reflection=0.2 (total 0.75) |
| region(s) | left_eye, nose, face_boundary |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | curved (rigidity 0.8, coverage 0.85) |
| illumination | mixed |
| recipe identity | `d4ce70aa0fc8ba4f078cff94996ab5a1e853578c54691c284971e409c9b76b57` |
| graph hash | `8d1af0e1537d5ad46f950b2c4ea5bf01449025f8b2b9dc600465cf8d8c550829` |

*Physical reading:* an emissive display panel (curved, transparency 0.05, roughness 0.15) covering left_eye, nose, face_boundary at coverage 0.85, producing moire at 0.25, pixel_grid at 0.3, specular_reflection at 0.2, captured under mixed lighting at yaw -12.5 deg, scale 1.05, compression q85.

### pilot_005

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000005 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.35, moire=0.25, specular_reflection=0.2 (total 0.8) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `aa26bceb7ca227b0351168bdd8f8443aaa7ac775f742cca77f6a418d98cc661f` |
| graph hash | `7e8aeb2b08f5318c2a79642248aed67892dfcf2153d99b03c8e521291361c079` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.35, moire at 0.25, specular_reflection at 0.2, captured under frontal light at yaw 12.5 deg, scale 1.0, compression q85.

### pilot_006

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000006 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | front |
| recipe identity | `61809526475563739564e535f241e9d06156456268167acf6170069c91cc1a62` |
| graph hash | `ecb9c0a900d385783784d9fc3503e175bfa62df9ea9e8d4590db8044b918781b` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw -12.5 deg, scale 1.05, compression q85.

### pilot_007

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000007 |
| artifact type(s) | moire, specular_reflection, pixel_grid |
| strength / severity | moire=0.25, specular_reflection=0.2, pixel_grid=0.3 (total 0.75) |
| region(s) | left_eye, nose, face_boundary |
| medium | display-like (transparency 0.02, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | front |
| recipe identity | `d67096bb1f2f4e0a03eef3bb1b7052510a678c2d588f4b8d917c8fdde6447f41` |
| graph hash | `37c5baa0ead4b16ac1bdd60d30a1df62088bad2aad8d4c293978a4ac405b0a83` |

*Physical reading:* an emissive display panel (held flat, transparency 0.02, roughness 0.1) covering left_eye, nose, face_boundary at coverage 0.85, producing moire at 0.25, specular_reflection at 0.2, pixel_grid at 0.3, captured under frontal light at yaw -5.5 deg, scale 1.02, compression q85.

### pilot_008

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000008 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.3, specular_reflection=0.2 (total 0.75) |
| region(s) | left_eye, nose, mouth |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `48e174ad1d3b748883ac2d339dc0aa9f2addae187b4278ed00dd6831ca739a51` |
| graph hash | `ed552ff4e65b5506a56aa1e272af5cd595bec44faebf818aacd1e7126829ea32` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.15) covering left_eye, nose, mouth at coverage 0.85, producing pixel_grid at 0.25, moire at 0.3, specular_reflection at 0.2, captured under frontal light at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_009

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000009 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.02, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | front |
| recipe identity | `8ff5db904a5020171d0c1ac4d992ec2da609043fe9a7737c5e458bcaeff273dd` |
| graph hash | `fe5b2672ea415f8605c89133ad287270d7390b97d0c76c421ffa6c7dd72e766c` |

*Physical reading:* an emissive display panel (held flat, transparency 0.02, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw 12.5 deg, scale 1.02, compression q85.

### pilot_010

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000010 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.35, moire=0.25, specular_reflection=0.15 (total 0.75) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | flat (rigidity 0.9, coverage 0.8) |
| illumination | front |
| recipe identity | `2c33ffc0f0e494fef5c64f8f5441f0b95a8c6d10b5ead126f892a85d26a7c9e1` |
| graph hash | `262d274dd4517a916b491365febe16b16a725b0fe2b01510f64b870083615717` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.15) covering left_eye, right_eye, nose at coverage 0.8, producing pixel_grid at 0.35, moire at 0.25, specular_reflection at 0.15, captured under frontal light at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_011

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000011 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | curved (rigidity 0.8, coverage 0.65) |
| illumination | mixed |
| recipe identity | `e51dcda13d185379c852bcfe7dd04598a963d9349e89193813bde3777df0f5de` |
| graph hash | `b9cf6f7ed27f63a87856e3850533e13f9994a3218293c2d98a98d35d2b80e5b1` |

*Physical reading:* an emissive display panel (curved, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.65, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under mixed lighting at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_012

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000012 |
| artifact type(s) | halftone, texture_smoothing, color_shift |
| strength / severity | halftone=0.25, texture_smoothing=0.3, color_shift=0.15 (total 0.7) |
| region(s) | left_eye, right_eye, nose |
| medium | paper-like (transparency 0.05, roughness 0.75) |
| geometry | curved (rigidity 0.35, coverage 0.85) |
| illumination | front |
| recipe identity | `07b7dcf98dae193de4e6f8e9f6395f0a96ec036583f8d4b8ffc3f32624225509` |
| graph hash | `90f3a262be51adb151c850589925c8cd9c66a7e4aff7b83a99c0c09abee83519` |

*Physical reading:* a printed paper surface (curved, transparency 0.05, roughness 0.75) covering left_eye, right_eye, nose at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.3, color_shift at 0.15, captured under frontal light at yaw -12.5 deg, scale 1.05, compression q85.

### pilot_013

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000013 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.3, moire=0.25 (total 0.55) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| recipe identity | `781721c119a8dea68e08d13fdaedc21359b27a4e905b2e994bde7d70d4b5126c` |
| graph hash | `d3f86cfcb7d6a6f0fa2168da596501632e54a86f4c599e62bee2b6bd37147380` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.3, moire at 0.25, captured under frontal light at yaw 0.0 deg, scale 1.0, compression q85.

### pilot_014

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000014 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, nose, mouth |
| medium | display-like (transparency 0.0, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | mixed |
| recipe identity | `8efbfb574d5a49f1e71901f9e883fbb068463bced970477998644b10754939a9` |
| graph hash | `bb0c463374d0f98baf2690b5aa96cab42f8a0f35a3c8d0909b11be16c5cc3064` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.1) covering left_eye, nose, mouth at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under mixed lighting at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_015

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000015 |
| artifact type(s) | halftone, texture_smoothing, color_shift |
| strength / severity | halftone=0.25, texture_smoothing=0.2, color_shift=0.15 (total 0.6) |
| region(s) | left_eye, nose, forehead |
| medium | paper-like (transparency 0.05, roughness 0.7) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| recipe identity | `bfbb9bf6bb8108f4b29200fb1e3a79e80b6362f6080d6fe1d4b50421b6e117f1` |
| graph hash | `7bbbd29d0a35037bc29aea5704110be8625ad0554b848ed91fce2c5cb59c71f6` |

*Physical reading:* a printed paper surface (held flat, transparency 0.05, roughness 0.7) covering left_eye, nose, forehead at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, color_shift at 0.15, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_016

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000016 |
| artifact type(s) | pixel_grid, specular_reflection |
| strength / severity | pixel_grid=0.25, specular_reflection=0.2 (total 0.45) |
| region(s) | left_eye, nose, forehead |
| medium | display-like (transparency 0.0, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `fb6c127e4942c74cd6905c0bf7ec62a8cea5dd073b0f96dbf4c454ce46932e1c` |
| graph hash | `76939d126617e1d86f0fc88ff1e55a8a27af1de538d67acc102491d571b85182` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.1) covering left_eye, nose, forehead at coverage 0.85, producing pixel_grid at 0.25, specular_reflection at 0.2, captured under frontal light at yaw 12.5 deg, scale 1.0, compression q85.

### pilot_017

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000017 |
| artifact type(s) | moire, pixel_grid, specular_reflection |
| strength / severity | moire=0.25, pixel_grid=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, forehead |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | curved (rigidity 0.8, coverage 0.85) |
| illumination | mixed |
| recipe identity | `5f3d46d41572cc4fee676852376c419bee3f8bc875793681bb1270cf3b8d9715` |
| graph hash | `832d24ebf5919f9f00b5057ce1ede4b2d12263e828eb060cad3c90a71b753bf6` |

*Physical reading:* an emissive display panel (curved, transparency 0.05, roughness 0.1) covering left_eye, right_eye, forehead at coverage 0.85, producing moire at 0.25, pixel_grid at 0.2, specular_reflection at 0.15, captured under mixed lighting at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_018

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000018 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, nose, face_boundary |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | flat (rigidity 0.85, coverage 0.9) |
| illumination | front |
| recipe identity | `750a249dae766bf55ca90e6e48a78d3f9ef3458223ed06923d2b65dd90b795a3` |
| graph hash | `3abb0bc2f45bb60d2caedb49b297928b7d9a522702a1ebac09281fa785ca7e21` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.15) covering left_eye, nose, face_boundary at coverage 0.9, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw 5.5 deg, scale 1.02, compression q85.

### pilot_019

| field | value |
| --- | --- |
| attempts (provider calls) | 2 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000019 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `f779d65f9ce4a4da36ddf01cd32cc17fa3600452382a565b0a9892e972bf9546` |
| graph hash | `675f6fc295572fb12d872cd002620313c9e215bdd306de3c3e5d326cfc58e7a0` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw -12.5 deg, scale 1.05, compression q85.

### pilot_020

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000020 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.35, moire=0.25, specular_reflection=0.2 (total 0.8) |
| region(s) | left_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | mixed |
| recipe identity | `3c04ee380c4fa117fd8a0eeb77a23791b85c70c4be1eb3593a6228c3708bd051` |
| graph hash | `9b21acd885e81ec8583924badb1a78b45c64c6be108ac64f23ca3bbac3f52ad8` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, nose at coverage 0.85, producing pixel_grid at 0.35, moire at 0.25, specular_reflection at 0.2, captured under mixed lighting at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_021

| field | value |
| --- | --- |
| attempts (provider calls) | 6 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000021 |
| artifact type(s) | halftone, texture_smoothing, color_shift |
| strength / severity | halftone=0.35, texture_smoothing=0.25, color_shift=0.15 (total 0.75) |
| region(s) | left_eye, nose, forehead |
| medium | paper-like (transparency 0.05, roughness 0.7) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| recipe identity | `3bfc2acd8fa58fe41fa5c67e5ad03708da8c1f98905d2f9610944ff28eba974a` |
| graph hash | `76aa6fbc98d5b74c5ab108f665e4d6cb4b2fd220725838e6d5817e6a32ac34c8` |

*Physical reading:* a printed paper surface (held flat, transparency 0.05, roughness 0.7) covering left_eye, nose, forehead at coverage 0.85, producing halftone at 0.35, texture_smoothing at 0.25, color_shift at 0.15, captured under frontal light at yaw -12.5 deg, scale 0.98, compression q85.

### pilot_022

| field | value |
| --- | --- |
| attempts (provider calls) | 5 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000022 |
| artifact type(s) | moire, pixel_grid, specular_reflection |
| strength / severity | moire=0.25, pixel_grid=0.3, specular_reflection=0.2 (total 0.75) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | curved (rigidity 0.8, coverage 0.85) |
| illumination | mixed |
| recipe identity | `04b6e45d11cec7e3d080aa64cc0b86abc308a1196d75025b98c6fd25f871bf92` |
| graph hash | `5a8231eddc567ffcaab86ccc3ceb1316ead486d4b021f35c4c90711386c3b6d0` |

*Physical reading:* an emissive display panel (curved, transparency 0.05, roughness 0.15) covering left_eye, right_eye, nose at coverage 0.85, producing moire at 0.25, pixel_grid at 0.3, specular_reflection at 0.2, captured under mixed lighting at yaw 12.5 deg, scale 1.05, compression q85.

### pilot_023

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000023 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | front |
| recipe identity | `6238fb550518f4d3238454c355137d15fd21b450d5012abcc6aa8355669b4cc6` |
| graph hash | `2fef8c93bfa99d316d70ca61a98651ab2e1fbb063ea70ea013fb77b6d2da9186` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_024

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000024 |
| artifact type(s) | pixel_grid, moire |
| strength / severity | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `2c3d58d4b262fa47dd0d6e95b7085376592d58087c7eb76d6eb2b92423c3d244` |
| graph hash | `ce2e5975d9c6396bc26e2ad9a3552ffdd3c0925143259aa22d0705cd395d9e4a` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, captured under frontal light at yaw -5.0 deg, scale 1.0, compression q85.

### pilot_025

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000025 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.3, moire=0.25, specular_reflection=0.2 (total 0.75) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `89d3df14a01c573a70947f659f4d822d37059c9e2dd765d559473b11eaffefd1` |
| graph hash | `855c1ba953461ed0e630cceab0a134a0c4af0276c2f85310e76348aa612f24ff` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.3, moire at 0.25, specular_reflection at 0.2, captured under frontal light at yaw 12.0 deg, scale 1.0, compression q85.

### pilot_026

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000026 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.8) |
| illumination | front |
| recipe identity | `a895091baf83b74941ab0f54bfe6136465729f529a8932a8e3f417dcd1f9cf49` |
| graph hash | `b208d4f00a4ec921b2edcb089d22b1d402e8dcb9b980d3c5c3061cadd16257f3` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.8, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_027

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000027 |
| artifact type(s) | halftone, texture_smoothing, color_shift |
| strength / severity | halftone=0.25, texture_smoothing=0.2, color_shift=0.15 (total 0.6) |
| region(s) | left_eye, nose, mouth |
| medium | paper-like (transparency 0.0, roughness 0.75) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| recipe identity | `a1857d0956aa310e92cb7fb1de84cfb63e50e794b948aa0497de40589af87039` |
| graph hash | `f5a046e4258e3bdb87c0e7fe79575f0aa63671dbf980bb81957dcf403409b399` |

*Physical reading:* a printed paper surface (held flat, transparency 0.0, roughness 0.75) covering left_eye, nose, mouth at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, color_shift at 0.15, captured under frontal light at yaw 12.5 deg, scale 1.0, compression q85.

### pilot_028

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000028 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, nose, mouth |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `6e5111813e50ddd1a6cdb8081475bc6a65e3998c826c2244e988b9e140c137a5` |
| graph hash | `c3949f6ac932b3a95ce2967fbb5130d5962beebc99390bb0bcbf6320b576ff47` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, nose, mouth at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw -12.5 deg, scale 1.02, compression q85.

### pilot_029

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000029 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.95, coverage 0.85) |
| illumination | front |
| recipe identity | `3c5afcdeec5389d23985418a664a69d10a7a5512db4da1404dcbe54c090bce70` |
| graph hash | `2a2f9b854f87a83f3767aec6026769339c56dadeccc27be4facde7b1c8e17d64` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw -5.5 deg, scale 1.02, compression q85.

### pilot_030

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000030 |
| artifact type(s) | halftone, texture_smoothing, color_shift |
| strength / severity | halftone=0.25, texture_smoothing=0.2, color_shift=0.15 (total 0.6) |
| region(s) | left_eye, nose, mouth |
| medium | paper-like (transparency 0.0, roughness 0.75) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `1b6ecd11dc2cd637b41b29c852f8d16675292488ff1285faa9a408fd9e655f82` |
| graph hash | `34f60f20d961b2ebef5ac32b1f5004e1d04006a7d6447eba7bd7bf28c8e618fd` |

*Physical reading:* a printed paper surface (held flat, transparency 0.0, roughness 0.75) covering left_eye, nose, mouth at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, color_shift at 0.15, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

### pilot_031

| field | value |
| --- | --- |
| attempts (provider calls) | 1 |
| final status | accepted |
| validation | accepted |
| compiler | compiled |
| recipe id | R-000031 |
| artifact type(s) | pixel_grid, moire, specular_reflection |
| strength / severity | pixel_grid=0.25, moire=0.2, specular_reflection=0.15 (total 0.6) |
| region(s) | left_eye, right_eye, nose |
| medium | display-like (transparency 0.05, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| recipe identity | `ed4c5c152fce2e846fe0f1be6c0645ab93f4ca479069e3e172b193e39bb13614` |
| graph hash | `093438507efe24a93852d3715a644328a0c6200849ca9f0b337a661b0b79d9bd` |

*Physical reading:* an emissive display panel (held flat, transparency 0.05, roughness 0.1) covering left_eye, right_eye, nose at coverage 0.85, producing pixel_grid at 0.25, moire at 0.2, specular_reflection at 0.15, captured under frontal light at yaw 5.0 deg, scale 1.0, compression q85.

