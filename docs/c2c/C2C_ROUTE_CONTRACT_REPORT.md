# C2C - scientific route contract repair

C2B produced 32 schema-valid recipes and then could not compile 10 of them, because
they declared `generator_route` without `physics`. The validator and the compiler
disagreed about what an acceptable recipe is. C2C resolves that in favour of the
synthesis design: `generator_route` is a frozen execution contract, and the only
accepted declaration is exactly `["physics", "gpat"]`.

**Outcome: PASS**

## The route policy

| field | value |
| --- | --- |
| version | prism_c_route_policy_v1 |
| required generator_route | `['physics', 'gpat']` |
| require exact order | True |
| subset allowed | False |
| GPAT-only accepted class | False |
| silent repair permitted | False |
| **route policy identity** | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` |

Canonical text: `{"allow_gpat_only_class":false,"allow_subset":false,"allowed_scientific_generator_route":["physics","gpat"],"require_exact_order":true,"silent_repair_permitted":false,"version":"prism_c_route_policy_v1"}`

> Physics is mandatory and every scientific recipe receives 4 physics + 4 GPAT candidate renders, so the RND/DET/LLM arms must share identical route exposure. A recipe that only one route can render would confound that comparison.

Enforcement sits in the validation pipeline, between the inherited validator's pass
and canonicalization. A route-invalid candidate is rejected as `rejected_route_policy`,
is never canonicalized, never registered in the duplicate registry, and is never
handed to the compiler. **Nothing is repaired**: a recipe declaring `["gpat"]` is
recorded exactly as the provider wrote it.

## Prompt amendment

| field | value |
| --- | --- |
| reason | C2B_VALIDATOR_COMPILER_ROUTE_CONTRACT_REPAIR |
| old prompt identity | `d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f` |
| new prompt identity | `e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737` |
| old system-prompt sha256 | `d0b4aaf1ba92c379a1bb707f7a1d36e483594dd200c2788f0c32ed2f6702ab9d` |
| new system-prompt sha256 | `a76da24194d4d0ce165298fffd8985f7e94290da5955a830816c62264997ff06` |
| generation template changed | False |
| coverage quotas changed | False |
| characters added | 410 |
| lines added / removed | 8 / 0 |

> source-independent engineering / spec reconciliation, NOT target tuning. No dataset, metric, attack family or target result was consulted; the amendment states an execution contract the frozen Version-C synthesis design already required.

Exact byte-level diff:

```diff
--- system_instruction (C1/C2/C2B)
+++ system_instruction (C2C)
@@ -35,4 +35,12 @@
 11. Never make one artifact a universal shortcut. If a recipe declares a
     forbidden shortcut, that recipe must not consist solely of that artifact.
+
+MANDATORY ROUTE DECLARATION
+12. "generator_route" is a fixed execution contract, NOT a diversity axis. Every
+    recipe must declare exactly:
+      "generator_route": ["physics", "gpat"]
+    Do not emit a physics-only route, a gpat-only route, a different order or any
+    other route set. Every recipe must be executable through both routes. A
+    recipe declaring anything else is invalid and is discarded.
 
 SCOPE RULE
```

## C2B replay under the new policy (offline, zero network)

| measure | as C2B ran it | under the C2C route policy |
| --- | --- | --- |
| accepted | 32 | 6 |
| rejected | 0 | 26 |
| rejected by route policy | 0 | 26 |
| compiler attempted | 32 | 6 |
| compiler compiled | 22 | 6 |
| **compiler failed** | 10 | 0 |

- no recipe was altered: **True**
- silent repairs: **0**
- C2B artifacts were read only and were not modified.

The observed split is reported as measured. It is **not** the 22/10 an initial reading
might expect: the frozen policy rejects physics-only as well as gpat-only, and C2B's
route distribution was 16 physics-only, 10 gpat-only and only 6 physics+gpat. So 6
recipes comply and 26 do not, and the compiler is never offered a recipe it cannot
build.

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
| accepted | 32 |
| route-policy rejections | 0 |
| other rejections | 0 |
| duplicates | 0 |
| compiled | 32 |
| **compiler failures among accepted** | 0 |

| provenance field | value |
| --- | --- |
| latency (s) | 77.354928 |
| model revision | gemini-3.6-flash |
| finish reason | completed |
| raw response sha256 | `f63568124983553e7f12b1d07824aa7e6c8fe63033f084f521dc8d85b9625b2c` |
| request sha256 | `e2c225b01f27c7b8715a5802772a05d77ef7209d2cb23ff51cae3a0eaec921cd` |
| route policy identity in request | `209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7` |
| total_cached_tokens | 0 |
| total_input_tokens | 1954 |
| total_output_tokens | 9496 |
| total_thought_tokens | 9503 |
| total_tokens | 20953 |
| total_tool_use_tokens | 0 |

- offline replay identical to the live run: **True**

## Route compliance of the live batch

| generator_route | recipes |
| --- | --- |
| physics+gpat | 32 |

- accepted recipes compliant with the contract: **32/32**
- silent repairs: **0** · GPAT-only class created: **False**

## Coverage and quota compliance

### artifacts - 8/8 present, max share 15.1515% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| halftone | 7 | 21.875 | 1 | 4 | 16 | pass |
| pixel_grid | 6 | 18.75 | 1 | 4 | 16 | pass |
| moire | 5 | 15.625 | 1 | 4 | 16 | pass |
| specular_reflection | 9 | 28.125 | 1 | 4 | 16 | pass |
| texture_smoothing | 10 | 31.25 | 1 | 4 | 16 | pass |
| color_shift | 9 | 28.125 | 1 | 4 | 16 | pass |
| boundary_inconsistency | 10 | 31.25 | 1 | 4 | 16 | pass |
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
| paper-like | 7 | 21.875 | 4 | - | 10 | pass |
| display-like | 7 | 21.875 | 4 | - | 10 | pass |
| plastic-like | 6 | 18.75 | 4 | - | 10 | pass |
| fabric-like | 6 | 18.75 | 4 | - | 10 | pass |
| reflective-film-like | 6 | 18.75 | 4 | - | 10 | pass |

### regions - 9/9 present, max share 14.1176% (PASS)

| category | count | % of recipes | min | preferred | max | required |
| --- | --- | --- | --- | --- | --- | --- |
| left_eye | 9 | 28.125 | 1 | 3 | 16 | pass |
| right_eye | 10 | 31.25 | 1 | 3 | 16 | pass |
| nose | 10 | 31.25 | 1 | 3 | 16 | pass |
| mouth | 8 | 25.0 | 1 | 3 | 16 | pass |
| forehead | 9 | 28.125 | 1 | 3 | 16 | pass |
| left_cheek | 8 | 25.0 | 1 | 3 | 16 | pass |
| right_cheek | 8 | 25.0 | 1 | 3 | 16 | pass |
| face_boundary | 12 | 37.5 | 1 | 3 | 16 | pass |
| context | 11 | 34.375 | 1 | 3 | 16 | pass |

| measure | min | max | mean | histogram |
| --- | --- | --- | --- | --- |
| artifacts per recipe | 2 | 3 | 2.0625 | {"2": 30, "3": 2} |
| regions per recipe | 1 | 3 | 2.65625 | {"1": 1, "2": 9, "3": 22} |

## Did the route fix damage coverage?

| axis | categories | C2B present | C2C present | C2B max share | C2C max share | still fully covered |
| --- | --- | --- | --- | --- | --- | --- |
| media | 5 | 5 | 5 | 21.875% | 21.875% | yes |
| geometry | 6 | 6 | 6 | 18.75% | 18.75% | yes |
| illumination | 6 | 6 | 6 | 18.75% | 18.75% | yes |
| artifacts | 8 | 8 | 8 | 16.4179% | 15.1515% | yes |
| regions | 9 | 9 | 9 | 12.1622% | 14.1176% | yes |

**coverage preserved.** The quota values were not changed: the same C2B generic
quotas were used unmodified.

## Co-occurrence

### artifacts_x_media

| artifacts | paper-like | display-like | plastic-like | fabric-like | reflective-film-like | total |
| --- | --- | --- | --- | --- | --- | --- |
| halftone | 5 | 0 | 0 | 2 | 0 | 7 |
| pixel_grid | 0 | 6 | 0 | 0 | 0 | 6 |
| moire | 0 | 3 | 0 | 0 | 2 | 5 |
| specular_reflection | 0 | 1 | 3 | 0 | 5 | 9 |
| texture_smoothing | 3 | 0 | 1 | 5 | 1 | 10 |
| color_shift | 2 | 1 | 3 | 2 | 1 | 9 |
| boundary_inconsistency | 2 | 2 | 3 | 1 | 2 | 10 |
| blur | 3 | 1 | 2 | 3 | 1 | 10 |

occupied cells: 27/40 · dominant: pixel_grid x display-like (6, 9.0909%)

### artifacts_x_geometry

| artifacts | flat | curved | partial-curved | flexible | rigid | boundary-only | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 2 | 2 | 0 | 1 | 1 | 1 | 7 |
| pixel_grid | 1 | 0 | 1 | 1 | 2 | 1 | 6 |
| moire | 1 | 1 | 0 | 2 | 1 | 0 | 5 |
| specular_reflection | 2 | 2 | 1 | 0 | 2 | 2 | 9 |
| texture_smoothing | 2 | 3 | 3 | 1 | 1 | 0 | 10 |
| color_shift | 2 | 0 | 0 | 4 | 1 | 2 | 9 |
| boundary_inconsistency | 0 | 1 | 2 | 0 | 1 | 6 | 10 |
| blur | 2 | 1 | 3 | 2 | 2 | 0 | 10 |

occupied cells: 37/48 · dominant: boundary_inconsistency x boundary-only (6, 9.0909%)

### artifacts_x_regions

| artifacts | left_eye | right_eye | nose | mouth | forehead | left_cheek | right_cheek | face_boundary | context | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halftone | 2 | 4 | 3 | 2 | 3 | 1 | 1 | 2 | 2 | 20 |
| pixel_grid | 3 | 3 | 1 | 3 | 3 | 1 | 1 | 1 | 1 | 17 |
| moire | 2 | 1 | 3 | 2 | 2 | 2 | 0 | 1 | 1 | 14 |
| specular_reflection | 3 | 2 | 1 | 1 | 4 | 1 | 1 | 4 | 6 | 23 |
| texture_smoothing | 4 | 4 | 5 | 3 | 0 | 4 | 5 | 2 | 2 | 29 |
| color_shift | 2 | 3 | 3 | 3 | 2 | 2 | 3 | 3 | 2 | 23 |
| boundary_inconsistency | 1 | 0 | 1 | 0 | 1 | 2 | 1 | 10 | 6 | 22 |
| blur | 1 | 5 | 4 | 2 | 4 | 3 | 6 | 1 | 2 | 28 |

occupied cells: 68/72 · dominant: boundary_inconsistency x face_boundary (10, 5.6818%)

### media_x_geometry

| media | flat | curved | partial-curved | flexible | rigid | boundary-only | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-like | 2 | 1 | 1 | 1 | 1 | 1 | 7 |
| display-like | 1 | 1 | 1 | 1 | 2 | 1 | 7 |
| plastic-like | 1 | 1 | 1 | 1 | 0 | 2 | 6 |
| fabric-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| reflective-film-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |

occupied cells: 29/30 · dominant: plastic-like x boundary-only (2, 6.25%)

### media_x_illumination

| media | front | left | right | top | bottom | mixed | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-like | 2 | 1 | 1 | 1 | 1 | 1 | 7 |
| display-like | 1 | 2 | 1 | 1 | 1 | 1 | 7 |
| plastic-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| fabric-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| reflective-film-like | 1 | 1 | 1 | 1 | 1 | 1 | 6 |

occupied cells: 30/30 · dominant: paper-like x front (2, 6.25%)

### geometry_x_illumination

| geometry | front | left | right | top | bottom | mixed | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat | 2 | 0 | 1 | 1 | 0 | 2 | 6 |
| curved | 0 | 2 | 2 | 0 | 0 | 1 | 5 |
| partial-curved | 0 | 0 | 0 | 0 | 5 | 0 | 5 |
| flexible | 0 | 2 | 2 | 1 | 0 | 0 | 5 |
| rigid | 2 | 2 | 0 | 1 | 0 | 0 | 5 |
| boundary-only | 2 | 0 | 0 | 2 | 0 | 2 | 6 |

occupied cells: 17/36 · dominant: partial-curved x bottom (5, 15.625%)

## All 32 returned objects

### index 0

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000000 |
| artifact(s) | halftone, texture_smoothing |
| strengths | halftone=0.2, texture_smoothing=0.15 (total 0.35) |
| region(s) | left_eye, right_eye, nose |
| medium | paper-like (transparency 0.05, roughness 0.8) |
| geometry | flat (rigidity 0.9, coverage 0.85) |
| illumination | front |
| canonical identity | `8d3e49fac257000031605a147d1fa26f8549e792f2bb122600efb94130fbeb62` |
| graph hash | `3288fabe69fdbd830634a8b6472fc8b34a664ea6b6574a37e179d11eff8abbf8` |

*Physical reading:* a printed paper surface (held flat, transparency 0.05, roughness 0.8) covering left_eye, right_eye, nose at coverage 0.85, producing halftone at 0.2, texture_smoothing at 0.15, under frontal light at yaw -10.0 deg, scale 1.0, compression q85.

### index 1

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000001 |
| artifact(s) | pixel_grid, moire |
| strengths | pixel_grid=0.25, moire=0.2 (total 0.45) |
| region(s) | nose, mouth, forehead |
| medium | display-like (transparency 0.1, roughness 0.1) |
| geometry | rigid (rigidity 0.95, coverage 0.9) |
| illumination | left |
| canonical identity | `64cd38fc4f7199f8c37e23362bd159ab59b6d883589c7813db89f4a2dfc46c24` |
| graph hash | `7d3fb7d6ee920e6c9eb98d0abf0a4605ea2234a46df17c0bef86c6158a3a1be7` |

*Physical reading:* an emissive display panel (rigid, transparency 0.1, roughness 0.1) covering nose, mouth, forehead at coverage 0.9, producing pixel_grid at 0.25, moire at 0.2, under light from the left at yaw 5.0 deg, scale 1.05, compression q90.

### index 2

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000002 |
| artifact(s) | specular_reflection, blur |
| strengths | specular_reflection=0.3, blur=0.15 (total 0.45) |
| region(s) | forehead, left_cheek, right_cheek |
| medium | plastic-like (transparency 0.3, roughness 0.2) |
| geometry | curved (rigidity 0.8, coverage 0.75) |
| illumination | right |
| canonical identity | `5f185f355bec014db900d18b134a5ebf5ccda7e2f093353c21cd0e0388b5f518` |
| graph hash | `d24b68bf6767c10b1984cd89198fdd9cc2e665776f5bc7745437ff0510e03776` |

*Physical reading:* a moulded plastic surface (curved, transparency 0.3, roughness 0.2) covering forehead, left_cheek, right_cheek at coverage 0.75, producing specular_reflection at 0.3, blur at 0.15, under light from the right at yaw 15.0 deg, scale 0.95, compression q75.

### index 3

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000003 |
| artifact(s) | texture_smoothing, color_shift |
| strengths | texture_smoothing=0.25, color_shift=0.15 (total 0.4) |
| region(s) | left_cheek, right_cheek, face_boundary |
| medium | fabric-like (transparency 0.0, roughness 0.85) |
| geometry | flexible (rigidity 0.2, coverage 0.8) |
| illumination | top |
| canonical identity | `a5a80af0ebd934029bbe1a6e662f55c7192fbcb59c0d36066e3e55904638cfbc` |
| graph hash | `d9b782d222fc6554df65ccb0c559245151bd12acc2e97f95b3b2e30483f16b54` |

*Physical reading:* a woven fabric surface (flexible and deformable, transparency 0.0, roughness 0.85) covering left_cheek, right_cheek, face_boundary at coverage 0.8, producing texture_smoothing at 0.25, color_shift at 0.15, under light from above at yaw -20.0 deg, scale 1.1, compression q70.

### index 4

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000004 |
| artifact(s) | specular_reflection, boundary_inconsistency |
| strengths | specular_reflection=0.35, boundary_inconsistency=0.2 (total 0.55) |
| region(s) | face_boundary, context |
| medium | reflective-film-like (transparency 0.4, roughness 0.05) |
| geometry | partial-curved (rigidity 0.6, coverage 0.6) |
| illumination | bottom |
| canonical identity | `0865acd9d9254ca62b15d09a539d535305ee19bb33d3bcc13182baf4897a79ac` |
| graph hash | `197c12751e55664ab63148d253c23efe488a797f2ffebf46a235299c2ad1b31c` |

*Physical reading:* a reflective film overlay (partly curved, transparency 0.4, roughness 0.05) covering face_boundary, context at coverage 0.6, producing specular_reflection at 0.35, boundary_inconsistency at 0.2, under light from below at yaw 0.0 deg, scale 0.85, compression q80.

### index 5

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000005 |
| artifact(s) | boundary_inconsistency, color_shift |
| strengths | boundary_inconsistency=0.3, color_shift=0.1 (total 0.4) |
| region(s) | face_boundary, context |
| medium | paper-like (transparency 0.0, roughness 0.75) |
| geometry | boundary-only (rigidity 0.7, coverage 0.3) |
| illumination | mixed |
| canonical identity | `69b66c0d98dd48e2e5f6697b9694828922f20a222e1fec4c61c451536c3e793c` |
| graph hash | `17d0fbd66933e9e99b2547abb2dcc3c6ab8ccab27c6638c9c0cb869676497162` |

*Physical reading:* a printed paper surface (present only at the face edge, transparency 0.0, roughness 0.75) covering face_boundary, context at coverage 0.3, producing boundary_inconsistency at 0.3, color_shift at 0.1, under mixed lighting at yaw -5.0 deg, scale 1.0, compression q95.

### index 6

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000006 |
| artifact(s) | specular_reflection, pixel_grid |
| strengths | specular_reflection=0.2, pixel_grid=0.3 (total 0.5) |
| region(s) | left_eye, right_eye, forehead |
| medium | display-like (transparency 0.05, roughness 0.15) |
| geometry | rigid (rigidity 0.9, coverage 0.95) |
| illumination | front |
| canonical identity | `4674f21ce979fdea93a4a9d7278a55559c6b05e0478a3f0b118cbffb3677cefa` |
| graph hash | `0f606109bb1dda7a8b89aa12211979f1b13591331a742083fc1a7f4e161e08eb` |

*Physical reading:* an emissive display panel (rigid, transparency 0.05, roughness 0.15) covering left_eye, right_eye, forehead at coverage 0.95, producing specular_reflection at 0.2, pixel_grid at 0.3, under frontal light at yaw 25.0 deg, scale 1.15, compression q80.

### index 7

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000007 |
| artifact(s) | color_shift, blur |
| strengths | color_shift=0.2, blur=0.2 (total 0.4) |
| region(s) | nose, mouth |
| medium | plastic-like (transparency 0.2, roughness 0.3) |
| geometry | flexible (rigidity 0.3, coverage 0.7) |
| illumination | left |
| canonical identity | `57a40345d089324a36f2a521dacb48b49e6c66f7c4d96a387abea21a8e65e834` |
| graph hash | `3347907e457d735e176b5a0f9b73599c1dc6e1ec6d72394a5b6d0de83c8fae0b` |

*Physical reading:* a moulded plastic surface (flexible and deformable, transparency 0.2, roughness 0.3) covering nose, mouth at coverage 0.7, producing color_shift at 0.2, blur at 0.2, under light from the left at yaw -15.0 deg, scale 0.9, compression q60.

### index 8

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000008 |
| artifact(s) | halftone, texture_smoothing |
| strengths | halftone=0.25, texture_smoothing=0.2 (total 0.45) |
| region(s) | left_eye, right_eye, mouth |
| medium | fabric-like (transparency 0.1, roughness 0.9) |
| geometry | curved (rigidity 0.4, coverage 0.85) |
| illumination | right |
| canonical identity | `e8cb22e3eaa0a47e2a21c90bf8f49fe2127e39a0770ed5c4fa2c4b9b4b439d7c` |
| graph hash | `f93745b23447770ff841e3d43709f434272cdb1cf66e476c289cdaab4358576e` |

*Physical reading:* a woven fabric surface (curved, transparency 0.1, roughness 0.9) covering left_eye, right_eye, mouth at coverage 0.85, producing halftone at 0.25, texture_smoothing at 0.2, under light from the right at yaw 10.0 deg, scale 1.0, compression q85.

### index 9

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000009 |
| artifact(s) | moire, specular_reflection |
| strengths | moire=0.25, specular_reflection=0.3 (total 0.55) |
| region(s) | forehead, context |
| medium | reflective-film-like (transparency 0.5, roughness 0.1) |
| geometry | flat (rigidity 0.85, coverage 0.9) |
| illumination | top |
| canonical identity | `1372250f3bc2351a48fcdeefad3a964c8c73957727967909abd6691a06e21814` |
| graph hash | `125b594838792321ad5694e11b1235e542b4a4611604c06bc71d73dfbf81647b` |

*Physical reading:* a reflective film overlay (held flat, transparency 0.5, roughness 0.1) covering forehead, context at coverage 0.9, producing moire at 0.25, specular_reflection at 0.3, under light from above at yaw -30.0 deg, scale 1.2, compression q90.

### index 10

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000010 |
| artifact(s) | blur, texture_smoothing |
| strengths | blur=0.2, texture_smoothing=0.2 (total 0.4) |
| region(s) | left_cheek, right_cheek |
| medium | paper-like (transparency 0.1, roughness 0.7) |
| geometry | partial-curved (rigidity 0.5, coverage 0.65) |
| illumination | bottom |
| canonical identity | `6188146da990b7065511369ac1c6b4f08d0a5bef03297ed5c9ca9dc00234b80c` |
| graph hash | `985d873849d69e58320f2adc9ae054b32ffb377301c491f7183fcb493c5fe436` |

*Physical reading:* a printed paper surface (partly curved, transparency 0.1, roughness 0.7) covering left_cheek, right_cheek at coverage 0.65, producing blur at 0.2, texture_smoothing at 0.2, under light from below at yaw 0.0 deg, scale 0.8, compression q65.

### index 11

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000011 |
| artifact(s) | moire, boundary_inconsistency |
| strengths | moire=0.3, boundary_inconsistency=0.15 (total 0.45) |
| region(s) | left_eye, nose, face_boundary |
| medium | display-like (transparency 0.0, roughness 0.05) |
| geometry | curved (rigidity 0.95, coverage 0.8) |
| illumination | mixed |
| canonical identity | `eef4614099fc8c460d5ab8052b6c7fefbdf318f996388d3319a3174d3a173ff5` |
| graph hash | `88041a9e2d352893285e9745d126a65cdf92884193e83883964e50a8fba850e4` |

*Physical reading:* an emissive display panel (curved, transparency 0.0, roughness 0.05) covering left_eye, nose, face_boundary at coverage 0.8, producing moire at 0.3, boundary_inconsistency at 0.15, under mixed lighting at yaw 35.0 deg, scale 1.0, compression q100.

### index 12

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000012 |
| artifact(s) | boundary_inconsistency, specular_reflection |
| strengths | boundary_inconsistency=0.25, specular_reflection=0.15 (total 0.4) |
| region(s) | face_boundary, context |
| medium | plastic-like (transparency 0.15, roughness 0.25) |
| geometry | boundary-only (rigidity 0.8, coverage 0.25) |
| illumination | front |
| canonical identity | `d8e82a371f5e7a1ed0f91588ae6af33c7669b8d94ce4bef233d8360330fb5f1c` |
| graph hash | `b467df5675a764001943d09056bc4977d4fba62135a0f2b097c454d0229036ad` |

*Physical reading:* a moulded plastic surface (present only at the face edge, transparency 0.15, roughness 0.25) covering face_boundary, context at coverage 0.25, producing boundary_inconsistency at 0.25, specular_reflection at 0.15, under frontal light at yaw -10.0 deg, scale 1.05, compression q75.

### index 13

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000013 |
| artifact(s) | color_shift, blur, texture_smoothing |
| strengths | color_shift=0.2, blur=0.15, texture_smoothing=0.15 (total 0.5) |
| region(s) | right_eye, nose, right_cheek |
| medium | fabric-like (transparency 0.05, roughness 0.8) |
| geometry | rigid (rigidity 0.7, coverage 0.7) |
| illumination | left |
| canonical identity | `85477345bbdb27db04fede682314b8b367f16f6db08096397a9922f19674a73b` |
| graph hash | `86d5129d409db3c14b5e3f51c98d9750ba45380b483a24b0f9bf453f5e857fe6` |

*Physical reading:* a woven fabric surface (rigid, transparency 0.05, roughness 0.8) covering right_eye, nose, right_cheek at coverage 0.7, producing color_shift at 0.2, blur at 0.15, texture_smoothing at 0.15, under light from the left at yaw 20.0 deg, scale 0.95, compression q80.

### index 14

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000014 |
| artifact(s) | moire, color_shift |
| strengths | moire=0.2, color_shift=0.15 (total 0.35) |
| region(s) | nose, mouth, left_cheek |
| medium | reflective-film-like (transparency 0.3, roughness 0.15) |
| geometry | flexible (rigidity 0.15, coverage 0.75) |
| illumination | right |
| canonical identity | `8f15774ad00c51a5fb07d58acce1cb0a4b0a32913927050e61d1a5ab1c5b7da0` |
| graph hash | `8c4092f85d522f0d3141c9b2602521d055ca08db0f7f22010ce28f8b70da4d49` |

*Physical reading:* a reflective film overlay (flexible and deformable, transparency 0.3, roughness 0.15) covering nose, mouth, left_cheek at coverage 0.75, producing moire at 0.2, color_shift at 0.15, under light from the right at yaw -5.0 deg, scale 1.0, compression q85.

### index 15

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000015 |
| artifact(s) | halftone, boundary_inconsistency |
| strengths | halftone=0.3, boundary_inconsistency=0.2 (total 0.5) |
| region(s) | forehead, left_cheek, face_boundary |
| medium | paper-like (transparency 0.0, roughness 0.65) |
| geometry | rigid (rigidity 0.85, coverage 0.85) |
| illumination | top |
| canonical identity | `ff0c5b39132c1a69595809deb768b092ff307583f83ebf3f75fc6f4ee5cbcb9c` |
| graph hash | `9d4b44d42a60e0319203f5094808379d4b954722402a91a0926cc223d98baece` |

*Physical reading:* a printed paper surface (rigid, transparency 0.0, roughness 0.65) covering forehead, left_cheek, face_boundary at coverage 0.85, producing halftone at 0.3, boundary_inconsistency at 0.2, under light from above at yaw 0.0 deg, scale 1.1, compression q90.

### index 16

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000016 |
| artifact(s) | pixel_grid, blur |
| strengths | pixel_grid=0.35, blur=0.1 (total 0.45) |
| region(s) | right_eye, mouth, right_cheek |
| medium | display-like (transparency 0.0, roughness 0.1) |
| geometry | partial-curved (rigidity 0.75, coverage 0.85) |
| illumination | bottom |
| canonical identity | `ed1f37fca4d62ebe17f8aa3b417ce58b044b9c704ba8ed42c4e2e34fb5ff853f` |
| graph hash | `0372694a291ee749b71e8fc3964913dc92dc6986b0471d6a3e9d54ebefbf9bba` |

*Physical reading:* an emissive display panel (partly curved, transparency 0.0, roughness 0.1) covering right_eye, mouth, right_cheek at coverage 0.85, producing pixel_grid at 0.35, blur at 0.1, under light from below at yaw -25.0 deg, scale 0.85, compression q70.

### index 17

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000017 |
| artifact(s) | specular_reflection, color_shift |
| strengths | specular_reflection=0.4, color_shift=0.1 (total 0.5) |
| region(s) | left_eye, right_eye, context |
| medium | plastic-like (transparency 0.4, roughness 0.1) |
| geometry | flat (rigidity 0.9, coverage 0.9) |
| illumination | mixed |
| canonical identity | `6d122317a35208f57b3b4384013070a5ff50f51db69a7204cb492e9eb844fa09` |
| graph hash | `ce5383ea0451cf286fc61ba0d9cc27a1430bcf73224c71a4684fc1b702ea13df` |

*Physical reading:* a moulded plastic surface (held flat, transparency 0.4, roughness 0.1) covering left_eye, right_eye, context at coverage 0.9, producing specular_reflection at 0.4, color_shift at 0.1, under mixed lighting at yaw 10.0 deg, scale 1.0, compression q95.

### index 18

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000018 |
| artifact(s) | halftone, boundary_inconsistency |
| strengths | halftone=0.2, boundary_inconsistency=0.25 (total 0.45) |
| region(s) | face_boundary, context |
| medium | fabric-like (transparency 0.0, roughness 0.95) |
| geometry | boundary-only (rigidity 0.3, coverage 0.35) |
| illumination | front |
| canonical identity | `ac2ed49eb31622930ad8775c2e871a7a6fb4eb93f941967f07f2dc92f066d693` |
| graph hash | `1881f37a6d822df098f4bf2426b23e25b2161818e402df1d4ed8ae5f8896e87d` |

*Physical reading:* a woven fabric surface (present only at the face edge, transparency 0.0, roughness 0.95) covering face_boundary, context at coverage 0.35, producing halftone at 0.2, boundary_inconsistency at 0.25, under frontal light at yaw 15.0 deg, scale 1.0, compression q80.

### index 19

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000019 |
| artifact(s) | specular_reflection, texture_smoothing |
| strengths | specular_reflection=0.25, texture_smoothing=0.2 (total 0.45) |
| region(s) | left_eye, nose, mouth |
| medium | reflective-film-like (transparency 0.2, roughness 0.05) |
| geometry | curved (rigidity 0.8, coverage 0.8) |
| illumination | left |
| canonical identity | `dd8c0af26f03e6e751a0128208347675276af71f6f0c8cd8e2a354a7ee987b95` |
| graph hash | `62c0c33efc1c336707f31ce283df4e85c9b7f8894f9e8ac4adb462ec0cd96d19` |

*Physical reading:* a reflective film overlay (curved, transparency 0.2, roughness 0.05) covering left_eye, nose, mouth at coverage 0.8, producing specular_reflection at 0.25, texture_smoothing at 0.2, under light from the left at yaw -40.0 deg, scale 1.15, compression q85.

### index 20

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000020 |
| artifact(s) | halftone, color_shift, blur |
| strengths | halftone=0.15, color_shift=0.2, blur=0.1 (total 0.45) |
| region(s) | right_eye, forehead, right_cheek |
| medium | paper-like (transparency 0.05, roughness 0.85) |
| geometry | flexible (rigidity 0.1, coverage 0.9) |
| illumination | right |
| canonical identity | `26277090a88c97d93e36cb95f8416a76bef818735df993a9abf9a26f5f18f027` |
| graph hash | `f35d2f743d83e8487f90a9e19b36d1893a15cc475eee2725afe2d002f04801eb` |

*Physical reading:* a printed paper surface (flexible and deformable, transparency 0.05, roughness 0.85) covering right_eye, forehead, right_cheek at coverage 0.9, producing halftone at 0.15, color_shift at 0.2, blur at 0.1, under light from the right at yaw 5.0 deg, scale 0.9, compression q50.

### index 21

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000021 |
| artifact(s) | boundary_inconsistency, pixel_grid |
| strengths | boundary_inconsistency=0.3, pixel_grid=0.2 (total 0.5) |
| region(s) | face_boundary, context |
| medium | display-like (transparency 0.0, roughness 0.2) |
| geometry | boundary-only (rigidity 0.85, coverage 0.4) |
| illumination | top |
| canonical identity | `5ef1458e13d52a38ed2c99a02051531a7a89c3f0d9ab65177243c4827d91a69f` |
| graph hash | `25bbf1ff2138c09d0424325846c9f7709fa427eaa410977dcb7944bc757f224d` |

*Physical reading:* an emissive display panel (present only at the face edge, transparency 0.0, roughness 0.2) covering face_boundary, context at coverage 0.4, producing boundary_inconsistency at 0.3, pixel_grid at 0.2, under light from above at yaw -15.0 deg, scale 1.0, compression q90.

### index 22

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000022 |
| artifact(s) | texture_smoothing, boundary_inconsistency |
| strengths | texture_smoothing=0.3, boundary_inconsistency=0.2 (total 0.5) |
| region(s) | left_cheek, right_cheek, face_boundary |
| medium | plastic-like (transparency 0.25, roughness 0.35) |
| geometry | partial-curved (rigidity 0.65, coverage 0.7) |
| illumination | bottom |
| canonical identity | `8f11b79af1a55cb1a6cf231fb0fe798f25fa95f66773b5132fc34251f8a9d092` |
| graph hash | `bdbf9684df4c1ab612981329225ff316a9ded88fbf1f6c9cf465ee50dec79757` |

*Physical reading:* a moulded plastic surface (partly curved, transparency 0.25, roughness 0.35) covering left_cheek, right_cheek, face_boundary at coverage 0.7, producing texture_smoothing at 0.3, boundary_inconsistency at 0.2, under light from below at yaw 30.0 deg, scale 1.05, compression q75.

### index 23

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000023 |
| artifact(s) | blur, texture_smoothing |
| strengths | blur=0.25, texture_smoothing=0.15 (total 0.4) |
| region(s) | left_eye, right_eye, nose |
| medium | fabric-like (transparency 0.15, roughness 0.75) |
| geometry | flat (rigidity 0.6, coverage 0.8) |
| illumination | mixed |
| canonical identity | `2173d0993a51c7226ffb83591f4767cbd3baf13efa4724ab9d92b9264d6c320e` |
| graph hash | `34774c24110793cadd2d1ae96e923fe2b185bb553633858e482dbcb7c910da2f` |

*Physical reading:* a woven fabric surface (held flat, transparency 0.15, roughness 0.75) covering left_eye, right_eye, nose at coverage 0.8, producing blur at 0.25, texture_smoothing at 0.15, under mixed lighting at yaw 0.0 deg, scale 1.0, compression q85.

### index 24

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000024 |
| artifact(s) | specular_reflection, blur |
| strengths | specular_reflection=0.45, blur=0.1 (total 0.55) |
| region(s) | forehead, face_boundary, context |
| medium | reflective-film-like (transparency 0.6, roughness 0.0) |
| geometry | rigid (rigidity 0.9, coverage 0.95) |
| illumination | front |
| canonical identity | `ce429a43c99091a3870dbffb06e0c4a5dd08e07037a6e298b3b4ecbcb4a60780` |
| graph hash | `6d4109b0c8741e6c0d3a3498c5fab92579a25643765fdf1d06212ae482170c2e` |

*Physical reading:* a reflective film overlay (rigid, transparency 0.6, roughness 0.0) covering forehead, face_boundary, context at coverage 0.95, producing specular_reflection at 0.45, blur at 0.1, under frontal light at yaw -20.0 deg, scale 1.25, compression q100.

### index 25

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000025 |
| artifact(s) | halftone, texture_smoothing |
| strengths | halftone=0.2, texture_smoothing=0.25 (total 0.45) |
| region(s) | nose, mouth, context |
| medium | paper-like (transparency 0.0, roughness 0.8) |
| geometry | curved (rigidity 0.7, coverage 0.75) |
| illumination | left |
| canonical identity | `1c3aab5f325babe385a775d3e98849c726fa09436d8f5e6a5d1d6cb348a307b2` |
| graph hash | `57a462c04ef1ed31d7362a6389bb077c4f0fc4bd0ddb2bc36f142247106839aa` |

*Physical reading:* a printed paper surface (curved, transparency 0.0, roughness 0.8) covering nose, mouth, context at coverage 0.75, producing halftone at 0.2, texture_smoothing at 0.25, under light from the left at yaw 10.0 deg, scale 0.8, compression q80.

### index 26

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000026 |
| artifact(s) | pixel_grid, color_shift |
| strengths | pixel_grid=0.25, color_shift=0.2 (total 0.45) |
| region(s) | left_eye, mouth, forehead |
| medium | display-like (transparency 0.0, roughness 0.05) |
| geometry | flat (rigidity 0.9, coverage 0.9) |
| illumination | right |
| canonical identity | `86642714b83a5cfc085eb29a266cf82bf308be481110ee75159ff1a7fcf9adb8` |
| graph hash | `1b265a97ff5d1f7533d29140dbfae548c05b19aa628e36b47c39da072097fedc` |

*Physical reading:* an emissive display panel (held flat, transparency 0.0, roughness 0.05) covering left_eye, mouth, forehead at coverage 0.9, producing pixel_grid at 0.25, color_shift at 0.2, under light from the right at yaw -5.0 deg, scale 1.0, compression q95.

### index 27

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000027 |
| artifact(s) | boundary_inconsistency, color_shift |
| strengths | boundary_inconsistency=0.35, color_shift=0.1 (total 0.45) |
| region(s) | face_boundary |
| medium | plastic-like (transparency 0.1, roughness 0.15) |
| geometry | boundary-only (rigidity 0.95, coverage 0.2) |
| illumination | top |
| canonical identity | `4d1cc267813537efd99556d2a6f50eb129b2e49de6168290cffa863f2b655416` |
| graph hash | `9d3e22a6616f419d3748ca967f264aa55899dbce99ffe5f182b05bd6cb3b388f` |

*Physical reading:* a moulded plastic surface (present only at the face edge, transparency 0.1, roughness 0.15) covering face_boundary at coverage 0.2, producing boundary_inconsistency at 0.35, color_shift at 0.1, under light from above at yaw 20.0 deg, scale 1.1, compression q70.

### index 28

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000028 |
| artifact(s) | texture_smoothing, blur |
| strengths | texture_smoothing=0.3, blur=0.2 (total 0.5) |
| region(s) | left_cheek, right_cheek, context |
| medium | fabric-like (transparency 0.0, roughness 0.9) |
| geometry | partial-curved (rigidity 0.4, coverage 0.65) |
| illumination | bottom |
| canonical identity | `28e13f10ff11aa0cc666c3597e45908fb44c42a4de9ab29d388614625e115b9b` |
| graph hash | `3977d29b3dcaae28078966f887bab8b33b76ea52a68af1e1bd59af6712d69ebc` |

*Physical reading:* a woven fabric surface (partly curved, transparency 0.0, roughness 0.9) covering left_cheek, right_cheek, context at coverage 0.65, producing texture_smoothing at 0.3, blur at 0.2, under light from below at yaw -35.0 deg, scale 0.95, compression q60.

### index 29

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000029 |
| artifact(s) | specular_reflection, boundary_inconsistency |
| strengths | specular_reflection=0.3, boundary_inconsistency=0.2 (total 0.5) |
| region(s) | face_boundary, context |
| medium | reflective-film-like (transparency 0.35, roughness 0.1) |
| geometry | boundary-only (rigidity 0.75, coverage 0.3) |
| illumination | mixed |
| canonical identity | `e086fc645002ab794e6b9767ee267ca71213a9cf6ebbfea26be3a445d065f659` |
| graph hash | `150ca4fee3f4db8bbe3bcdb3edf5051353a01d90a0a3e271f3d5c079be3999e1` |

*Physical reading:* a reflective film overlay (present only at the face edge, transparency 0.35, roughness 0.1) covering face_boundary, context at coverage 0.3, producing specular_reflection at 0.3, boundary_inconsistency at 0.2, under mixed lighting at yaw 0.0 deg, scale 1.0, compression q85.

### index 30

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000030 |
| artifact(s) | halftone, blur |
| strengths | halftone=0.25, blur=0.15 (total 0.4) |
| region(s) | right_eye, nose, forehead |
| medium | paper-like (transparency 0.0, roughness 0.9) |
| geometry | flat (rigidity 0.8, coverage 0.85) |
| illumination | front |
| canonical identity | `33dca0cb73b012876c15edef23aec32a83d9e6069e1dee5bbb16972d393fa12f` |
| graph hash | `fa2aa17ef01be2b7c62ab0931d1dabd306f19468f8abc5d3d4738e0461314f63` |

*Physical reading:* a printed paper surface (held flat, transparency 0.0, roughness 0.9) covering right_eye, nose, forehead at coverage 0.85, producing halftone at 0.25, blur at 0.15, under frontal light at yaw 15.0 deg, scale 1.05, compression q90.

### index 31

| field | value |
| --- | --- |
| status | accepted |
| generator_route | ["physics", "gpat"] |
| compiler | compiled |
| recipe id | R-000031 |
| artifact(s) | moire, pixel_grid |
| strengths | moire=0.3, pixel_grid=0.2 (total 0.5) |
| region(s) | left_eye, right_eye, left_cheek |
| medium | display-like (transparency 0.0, roughness 0.1) |
| geometry | flexible (rigidity 0.2, coverage 0.75) |
| illumination | left |
| canonical identity | `04a4e7625cf9fdfddfafcc83d2ea887904086a67b5904817c6b5f76f8a8cb986` |
| graph hash | `fe4f77e11d93305cc05aae53cc88553b72858a9f88325bb10ba2d8a9e3fc5f5f` |

*Physical reading:* an emissive display panel (flexible and deformable, transparency 0.0, roughness 0.1) covering left_eye, right_eye, left_cheek at coverage 0.75, producing moire at 0.3, pixel_grid at 0.2, under light from the left at yaw -10.0 deg, scale 0.9, compression q85.

## Verdict detail

| check | result |
| --- | --- |
| returned_exactly_32 | pass |
| no_response_level_issues | pass |
| semantic_validity_at_least_threshold | pass |
| zero_compiler_failures_among_accepted | pass |
| duplicate_rate_within_threshold | pass |
| all_axes_fully_represented | pass |
| no_severe_mode_collapse | pass |
| quota_required_bounds_satisfied | pass |

Thresholds fixed before the batch returned: `{"returned_objects_exact": 32, "min_semantic_validity": 0.9, "max_response_issues": 0, "max_compiler_failures_among_accepted": 0, "max_duplicate_rate": 0.1, "max_axis_share_percent": 60.0, "axes_requiring_full_presence": ["media", "geometry", "illumination", "artifacts", "regions"]}`

These recipes are disposable validation evidence. None enters the C3 raw 384 slots,
the final LLM bank, an RND/DET bank, a synthetic bank or any training.

