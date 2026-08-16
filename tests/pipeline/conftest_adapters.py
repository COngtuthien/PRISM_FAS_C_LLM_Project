"""Shared helpers for the adapter tests. Imported, not collected.

A sandbox repository is the unit of isolation here: the adapters read real
configuration and real frozen evidence, so the tests copy those into a temp tree
and let the adapters write there. Nothing a test does can touch the committed
`reports/c3/` evidence.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: What an adapter needs on disk to run. `raw_responses` is excluded because it
#: is bulky provider payload and no adapter reads it during these tests.
_SANDBOX_TREES = ("configs", "docs", "reports", "scripts", "src")
_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "raw_responses",
                                  "validate", "smoke", "full")


def make_sandbox(target: Path) -> Path:
    """A copy of the repository the adapters can safely write into.

    `reports/c3/live/` is copied in full, raw archives included, even though
    `raw_responses` is otherwise skipped as bulk. A sandbox holding the C3 live
    state file without its archives is not a smaller repository — it is a
    corrupt one, and the resume integrity check correctly refuses it. Copying
    both keeps the sandbox coherent so tests exercise the gate they name rather
    than the drift detector.
    """
    for relative in _SANDBOX_TREES:
        source = REPO / relative
        if source.exists():
            shutil.copytree(source, target / relative, ignore=_IGNORED)

    live = REPO / "reports" / "c3" / "live"
    if live.exists():
        shutil.copytree(live, target / "reports" / "c3" / "live", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    from prism_fas.pipeline.adapters.context import reset_cache

    reset_cache()
    return target


def profile(name: str, repo: Path) -> Any:
    from prism_fas.pipeline.profiles import load_profile

    return load_profile(name, repo=repo)


def request_for(repo: Path, profile_name: str = "smoke", **kwargs: Any) -> Any:
    from prism_fas.pipeline.adapters import AdapterRequest

    return AdapterRequest(repo=repo, profile=profile(profile_name, repo), **kwargs)


def schedule_for(repo: Path) -> dict[str, int]:
    from prism_fas.pipeline.adapters.context import frozen_schedule

    return frozen_schedule(repo)


def mock_request(repo: Path, *, profile_name: str = "smoke", script: list[Any],
                 resume: bool = True, **kwargs: Any) -> Any:
    """An adapter request bound to a scripted mock provider."""
    from prism_fas.llm.providers.mock import MockRecipeProvider

    provider = MockRecipeProvider(script, model_id="test-mock")
    request = request_for(
        repo, profile_name, mode="LIVE_GENERATE", resume=resume,
        options={"mock_provider": provider, "sleep": lambda _s: None}, **kwargs)
    from prism_fas.pipeline.adapters import ProviderBinding

    return request, provider, ProviderBinding.MOCK


def valid_script(repo: Path, count: int, *, salt_base: int = 0) -> list[Any]:
    from prism_fas.pipeline.adapters.fixtures import scripted_success

    schedule = schedule_for(repo)
    return scripted_success(repo=repo, recipes=schedule["objects_per_request"],
                            count=count, salt_base=salt_base)


def live_state_path(repo: Path, profile_name: str = "smoke") -> Path:
    from prism_fas.pipeline.adapters.c3 import LIVE_STATE_FILE, live_dir_for

    return repo / live_dir_for(profile(profile_name, repo)) / LIVE_STATE_FILE


def run_c3(repo: Path, *, script: list[Any], profile_name: str = "smoke",
           resume: bool = True) -> Any:
    """Run the C3 live path once against a script, returning the result."""
    from prism_fas.pipeline.adapters.c3 import C3Mode, _live_generate

    request, provider, binding = mock_request(
        repo, profile_name=profile_name, script=script, resume=resume)
    result = _live_generate(request, C3Mode.LIVE_GENERATE, binding)
    result.detail["_provider"] = provider
    return result


def crash_after(repo: Path, completed: int, *, profile_name: str = "smoke",
                salt_base: int = 0) -> None:
    """Complete `completed` logical requests, then terminate abruptly.

    The mock raises `AssertionError` once its script runs out, which is as close
    as a test gets to the process being killed mid-request: the exception
    escapes the adapter without any cleanup path running. What survives is only
    what was already written to disk, which is exactly the property a resume
    depends on.
    """
    import pytest

    with pytest.raises(AssertionError, match="script exhausted"):
        run_c3(repo, script=valid_script(repo, completed, salt_base=salt_base),
               profile_name=profile_name)
