"""Fixtures for the C7 decision-contract tests.

The suite exercises the real detector on CPU fixtures. It loads no pretrained
weight, opens no dataset and reaches no network — the global tower is the same
shape-exact stub `variant_audit` already uses — so it needs nothing from the
environment beyond the repository itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO
