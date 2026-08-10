# C2 - prompt review

The full 32-slot pilot ran to completion under the unchanged C1 prompt. This review
was written afterwards, from the pilot evidence, and no prompt byte was changed
during C2.

## Rule applied

A prompt change may be recommended only on an objective, source-independent contract
issue: low schema compliance, a high duplicate rate, severe coverage collapse,
systematic ontology mismatch, or systematic physical incompatibility. A recipe that
merely looks unusual is not a reason, and nothing in this review consulted a dataset,
a target metric or an attack taxonomy.

## Measured against the thresholds

| criterion | observed | threshold | within contract |
| --- | --- | --- | --- |
| schema_compliance | 1.0 | >= 0.9 | yes |
| first_attempt_compliance | 1.0 | >= 0.75 | yes |
| duplicate_rate | 0.0 | <= 0.1 | yes |
| ontology_violation_rate | 0.0 | <= 0.05 | yes |
| compatibility_violation_rate | 0.0 | <= 0.1 | yes |
| coverage_geometry | 0.333333 | >= 0.5 | **NO** |
| mode_collapse | 3 | no category above 60.0% of its axis | **NO** |

- **schema_compliance** - share of slots that produced an accepted recipe within the frozen retry budget
- **first_attempt_compliance** - share of slots valid on the first call; a low value means the prompt under-constrains
- **duplicate_rate** - exact canonical-content repeats across the pilot
- **ontology_violation_rate** - non-canonical enum values per provider call; systematic mismatch would mean the vocabulary in the prompt does not match the validator
- **compatibility_violation_rate** - medium/artifact and geometry/region rejections per call
- **coverage_geometry** - weakest axis; missing ['partial-curved', 'flexible', 'rigid', 'boundary-only']
- **mode_collapse** - collapsed categories: [{'axis': 'media', 'category': 'display-like', 'share_percent': 81.25}, {'axis': 'geometry', 'category': 'flat', 'share_percent': 84.375}, {'axis': 'illumination', 'category': 'front', 'share_percent': 81.25}]

## Verdict

**A material contract issue was found.** Criteria breached: coverage_geometry, mode_collapse.

| axis | dominant category | share of axis |
| --- | --- | --- |
| media | display-like | 81.25% |
| geometry | flat | 84.375% |
| illumination | front | 81.25% |

### The measurement is real; the cause is not isolated

The coverage breach is measured, but its cause is not isolated. The frozen prompt's diversity rules are BATCH-scoped ('vary medium, geometry, region coverage ... within this batch'), and a C2 pilot slot asked for exactly one recipe, so those rules had no batch to act on. The pilot also passed no coverage quotas, although the request template supports them. The C3 schedule asks for 32 recipes per call, where both mechanisms do have scope.

- **What the evidence shows:** Under a batch of one, with no coverage quotas and no sampling controls, 32 independent calls under this prompt return a strongly modal distribution: display-like / flat / front dominate, and four of six geometries, three of five media and four of six illumination modes never appear.
- **What it does not show:** It does not establish that the prompt would collapse the same way at the C3 batch shape, and it is not evidence that any prompt wording is wrong.
- pilot batch size: 1; C3 planned batch size: 32; coverage quotas used: False

### Recommended next step

Before editing a single prompt byte, re-measure coverage at the C3 batch shape (one call requesting 32 recipes, the frozen 12x32 schema) so the batch-size confound is removed. Only if the collapse survives that does a prompt change have evidence behind it.

- If the collapse survives that: Use the request template's existing coverage_quotas mechanism, which is ontology-level and source-only, rather than rewording the system instruction. It is already implemented and was deliberately left unused in C2.

**No second 32-slot pilot was run and no prompt byte was changed.** C2 stops here
for user review, as instructed.

## Alias policy

`allow_ontology_aliases` remains **OFF**. It was not changed, and it is not proposed
for change: it is identity-bearing, so enabling it would alter the recipe bank
identity. Enabling it is a user decision only.

## Observed oddities that are NOT prompt defects

None were flagged.

