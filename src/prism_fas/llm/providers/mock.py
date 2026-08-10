"""Deterministic mock provider. Never touches the network.

Scripted responses are consumed in order, so a test can express an exact
sequence: transient failure, then success; or three invalid candidates, then a
retry-budget exhaustion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import ErrorClass, GenerationRequest, ProviderError, ProviderGenerationResult
from .base import RecipeProvider


@dataclass(frozen=True)
class ScriptedResponse:
    """One scripted provider outcome: either `raw_text` or an `error`."""

    raw_text: str | None = None
    error: ProviderError | None = None
    finish_reason: str | None = "stop"
    latency_seconds: float | None = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = "mock-model-version"
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        if (self.raw_text is None) == (self.error is None):
            raise ValueError("a scripted response carries exactly one of raw_text or error")


class MockRecipeProvider(RecipeProvider):
    """Replays a fixed script. Deterministic and offline."""

    name = "mock"

    def __init__(self, script: list[ScriptedResponse] | None = None, *,
                 model_id: str = "mock-model") -> None:
        self._script = list(script or [])
        self._model_id = model_id
        self.calls: list[tuple[str, int]] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    def _next(self) -> ScriptedResponse:
        if not self._script:
            raise AssertionError("MockRecipeProvider script exhausted: the code under test "
                                 "made more provider calls than the test scripted")
        return self._script.pop(0)

    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        self.calls.append((request.slot_id, attempt))
        scripted = self._next()
        return ProviderGenerationResult(
            slot_id=request.slot_id,
            attempt=attempt,
            provider=self.name,
            model_id=self._model_id,
            raw_text=scripted.raw_text,
            parsed=None,
            finish_reason=scripted.finish_reason,
            latency_seconds=scripted.latency_seconds,
            usage=dict(scripted.usage),
            provider_request_id=scripted.provider_request_id,
            model_version=scripted.model_version,
            provider_seed=None,
            error=scripted.error,
            sdk_version=None,
            api_surface="mock",
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model_id": self._model_id,
                "scripted_responses_remaining": self.remaining, "network": False}


def transport_error(message: str = "connection reset") -> ProviderError:
    return ProviderError(ErrorClass.TRANSPORT, message)


def rate_limit_error(retry_after_seconds: float = 1.0) -> ProviderError:
    return ProviderError(ErrorClass.RATE_LIMIT, "429 rate_limit_exceeded",
                         status_code=429, retry_after_seconds=retry_after_seconds)


def quota_exhausted_error() -> ProviderError:
    return ProviderError(ErrorClass.QUOTA_EXHAUSTED, "429 quota_exceeded: daily quota exhausted",
                         status_code=429)


def auth_error() -> ProviderError:
    return ProviderError(ErrorClass.AUTH, "401 API key not valid", status_code=401)


def model_unavailable_error(model_id: str) -> ProviderError:
    return ProviderError(ErrorClass.MODEL_UNAVAILABLE, f"404 model {model_id!r} is not available",
                         status_code=404)


def unsupported_config_error(detail: str) -> ProviderError:
    return ProviderError(ErrorClass.UNSUPPORTED_CONFIG, f"400 unsupported configuration: {detail}",
                         status_code=400)
