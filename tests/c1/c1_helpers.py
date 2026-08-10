"""Small helpers shared by the C1 test modules."""
from __future__ import annotations

import json
from typing import Any


def envelope(*candidates: dict[str, Any]) -> str:
    """The exact response envelope the provider contract requires."""
    return json.dumps({"recipes": list(candidates)})
