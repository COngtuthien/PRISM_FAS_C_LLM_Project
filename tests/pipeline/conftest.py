"""Fixtures for the v1.5 execution-layer tests. Offline only.

The execution layer is the thing that will eventually be allowed to spend live
provider quota and GPU hours, so its own test suite is the last place that
should be able to do either. Sockets are blocked and ambient credentials are
deleted for every test in this package, and the guards are autouse so a new test
cannot opt out by forgetting to ask for them.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
# The tests/ tree carries no __init__.py by convention, so the shared adapter
# helpers are imported by module name rather than as a package relative.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError(
            "an execution-layer test attempted a network connection; the pipeline package "
            "is built and tested entirely offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def validate_profile(repo: Path):
    from prism_fas.pipeline.profiles import load_profile

    return load_profile("validate", repo=repo)


@pytest.fixture(scope="session")
def full_profile(repo: Path):
    from prism_fas.pipeline.profiles import load_profile

    return load_profile("full", repo=repo)
