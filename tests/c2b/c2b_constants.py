"""Constants shared by the C2B test modules.

Kept out of `conftest.py` so the test modules can import them directly: pytest
puts this directory on `sys.path`, but `conftest` is not an importable package.
"""
from __future__ import annotations

BATCH_SIZE = 32
C2B_BANK_ID = "c2b-batch-disposable"

#: The single-recipe item schema identity C2 ran under. C2B must not move it.
FROZEN_ITEM_SCHEMA_IDENTITY = "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579"
FROZEN_ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"
FROZEN_SYSTEM_PROMPT_IDENTITY = "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f"

#: Recorded at C1 for the 12x32 schedule; the provider rejects it (see
#: reports/c2b/C2B_ENVELOPE_REJECTION.json).
C1_BOUNDED_BATCH_ENVELOPE_IDENTITY = (
    "7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4")
