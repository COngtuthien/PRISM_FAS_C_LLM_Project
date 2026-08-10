"""Constants shared by the C2C test modules."""
from __future__ import annotations

BATCH_SIZE = 32
C2C_BANK_ID = "c2c-batch-disposable"
C2B_BANK_ID = "c2b-batch-disposable"

#: The frozen scientific route contract.
REQUIRED_ROUTE = ["physics", "gpat"]
ROUTE_POLICY_IDENTITY = "209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7"

#: Unchanged by C2C: the route rule is enforced at the validation layer, not in
#: the JSON schema, so recipe semantics did not move.
FROZEN_ITEM_SCHEMA_IDENTITY = "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579"
FROZEN_ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"
FROZEN_BATCH_ENVELOPE_IDENTITY = (
    "f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113")

#: The prompt before and after the minimal route-contract amendment.
C2B_SYSTEM_PROMPT_IDENTITY = "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f"
C2C_SYSTEM_PROMPT_IDENTITY = "e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737"
