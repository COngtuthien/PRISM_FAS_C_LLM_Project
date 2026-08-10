# C2C - schema identity naming correction (prospective)

This note corrects a NAMING error going forward. It edits no historical artifact:
the C1, C2 and C2B reports are left exactly as they were recorded, including the
original wording, so the record of how the mistake arose stays intact.

## The error

C1 recorded a value under a name that reads as a single-recipe identity:

    llm_schema_identity_12x32 = 7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4

and later instructions referred to it as "the single-recipe schema identity". It is
not. It is the identity of the **32-object batch envelope**: the whole
`{"recipes": [...]}` object with `minItems = maxItems = 32`.

## The correct values

| schema | identity | what it actually is |
| --- | --- | --- |
| single-recipe ITEM schema | `1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579` | one recipe object; the thing that carries recipe semantics |
| C2 singleton envelope (n=1) | `e9f66067c2de2deda5373a99dc6c92689c0ab2d2163b80adcde57af83df9bbd1` | envelope C2 sent, accepted 42 times |
| C1-recorded 32-object envelope | `7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4` | envelope with the array bound; **rejected by the provider**, 400 INVALID_ARGUMENT |
| C2B/C2C batch envelope (sent) | `f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113` | same envelope without the array length bound |

## Why the distinction matters

The item schema is what recipe semantics depend on. The envelope only says how many
of those items a response carries. Conflating them made two separate facts look like
one:

- the envelope had to change, because the provider rejects the bounded form;
- the item schema did **not** change, and is byte-identical inside the 1-object
  envelope C2 used and the envelope C2B and C2C send.

C2C enforces the route contract at the validation layer rather than in the schema
precisely so the item identity stays put.

## Naming used from C2C onward

| name | meaning |
| --- | --- |
| `single_recipe_schema_identity` | the item schema, one recipe object |
| `batch_envelope_schema_identity` | the envelope actually sent |
| `bounded_batch_envelope_identity` | the envelope with array bounds, kept only to name what the provider refuses |

No historical report was edited to hide the original naming.

