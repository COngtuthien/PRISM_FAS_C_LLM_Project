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


@pytest.fixture(autouse=True)
def no_live_provider_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may build a real provider, whatever the gates decide.

    Blocked sockets already stop a request from leaving, but that is a backstop
    that fires *after* the code has decided to go live. This fixture makes the
    decision itself impossible to act on, so a test whose premise quietly stops
    holding — an absent quota snapshot that later exists, say — fails loudly at
    the construction point instead of silently exercising the live path.
    """
    import prism_fas.pipeline.adapters.c3 as c3

    original = c3._build_provider

    def guarded(binding, request):
        from prism_fas.pipeline.adapters import ProviderBinding

        if binding is ProviderBinding.LIVE:
            raise AssertionError(
                "a test reached live provider construction; every gate that should have "
                "stopped it is either satisfied or bypassed, and that is the bug")
        return original(binding, request)

    monkeypatch.setattr(c3, "_build_provider", guarded)


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
