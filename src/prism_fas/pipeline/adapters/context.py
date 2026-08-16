"""Access to the canonical C2C contract context.

`RouteContext` in `scripts/c2c_common.py` is the authority for the frozen
contract: it loads the ontology, provider config, coverage quotas, route policy
and prompt from disk, computes every identity from those bytes, and refuses to
construct itself if any of them has drifted from the accepted C2B/C2C freeze.
The C3 request shape — system instruction, rendered input text, batch schema,
model, thinking level, token budget — is built there too.

Reimplementing any of that here would create a second contract authority, so
this module imports it instead, and does nothing else. The `scripts/` path
insertion is the cost of that decision and is confined to this file; the
restructure plan relocates the module in a later step, at which point only this
import changes.

Constructing a `RouteContext` reads configuration files and computes hashes. It
opens no socket, reads no credential and makes no provider call, which is why
the validate profile can use it.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterError


class CanonicalContextUnavailable(AdapterError):
    """The canonical contract context could not be loaded.

    Raised rather than substituted. An adapter that cannot reach the real
    contract must block, because the alternative is inventing one.
    """


def _ensure_scripts_on_path(repo: Path) -> None:
    scripts = str((repo / "scripts").resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


@lru_cache(maxsize=4)
def _cached_context(repo_key: str) -> Any:
    repo = Path(repo_key)
    _ensure_scripts_on_path(repo)
    try:
        from c2c_common import RouteContext  # type: ignore[import-not-found]
    except ImportError as error:
        raise CanonicalContextUnavailable(
            "scripts/c2c_common.py does not import; it is the canonical C2C contract "
            f"context and there is no substitute for it ({error})") from error
    try:
        return RouteContext()
    except Exception as error:
        raise CanonicalContextUnavailable(
            "the canonical C2C contract context refused to construct, which means the "
            f"frozen contract has drifted: {type(error).__name__}: {error}") from error


def route_context(repo: Path) -> Any:
    """The live C2C contract context, loaded once per repository.

    Cached because constructing it hashes several config files and every C3
    adapter mode needs the same object; the cache key is the repository path so
    a test operating on a sandbox copy gets its own.
    """
    return _cached_context(str(Path(repo).resolve()))


@lru_cache(maxsize=4)
def _cached_constants(repo_key: str) -> dict[str, int]:
    repo = Path(repo_key)
    _ensure_scripts_on_path(repo)
    try:
        import c2c_common  # type: ignore[import-not-found]
    except ImportError as error:
        raise CanonicalContextUnavailable(
            f"scripts/c2c_common.py does not import ({error})") from error
    return {
        "requests": int(c2c_common.C3_REQUESTS),
        "objects_per_request": int(c2c_common.BATCH_SIZE),
        "raw_slots": int(c2c_common.C3_RAW_SLOTS),
        "minimum_unique_pool": int(c2c_common.C3_MIN_UNIQUE_POOL),
        "final_bank": int(c2c_common.C3_FINAL_BANK),
    }


def frozen_schedule(repo: Path) -> dict[str, int]:
    """The frozen C3 schedule, read from the canonical module.

    Not written here. These five numbers are scientific constants owned by the
    spec and the bank lock; this module reports them so an adapter can check
    itself against them rather than restate them.
    """
    return dict(_cached_constants(str(Path(repo).resolve())))


def reset_cache() -> None:
    """Drop the cached contexts. Used by tests that mutate a sandbox repo."""
    _cached_context.cache_clear()
    _cached_constants.cache_clear()


__all__ = ["CanonicalContextUnavailable", "route_context", "frozen_schedule", "reset_cache"]
