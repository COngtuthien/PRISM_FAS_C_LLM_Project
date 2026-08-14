"""Fixtures for the C3 BANK_LOCK tests. Offline only."""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

LOCK_PATH = REPO / "reports" / "c3" / "C3_BANK_LOCK.json"
VERIFICATION_PATH = REPO / "reports" / "c3" / "C3_BANK_LOCK_VERIFICATION.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError("a C3 bank-lock test attempted a network connection; the lock is "
                             "built and verified entirely offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="session")
def ontology():
    from prism_fas.recipes.ontology import load_ontology

    return load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")


@pytest.fixture(scope="session")
def route_policy():
    from prism_fas.llm.route_policy import load_route_policy

    return load_route_policy(REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml")


@pytest.fixture(scope="session")
def context():
    """The live C2C contract context, loaded from configuration on disk."""
    from c2c_common import RouteContext

    return RouteContext()


@pytest.fixture(scope="session")
def lock() -> dict[str, Any]:
    if not LOCK_PATH.exists():
        pytest.skip("C3_BANK_LOCK.json missing; run: python scripts/c3_bank_lock.py")
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def verification() -> dict[str, Any]:
    if not VERIFICATION_PATH.exists():
        pytest.skip("C3_BANK_LOCK_VERIFICATION.json missing; run scripts/c3_bank_lock.py")
    return json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
