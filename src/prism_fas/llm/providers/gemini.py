"""Google Gemini Developer API provider.

Verified against the installed SDK and the official documentation at C1; the
evidence is recorded in `reports/c1/C1_PROVIDER_AUDIT.json`.

    package  google-genai
    surface  client.interactions.create(...)   (GA; recommended for current models)
    model    gemini-3.6-flash                  (stable)
    thinking generation_config={"thinking_level": "medium"}
    output   response_format={"type": "text",
                              "mime_type": "application/json",
                              "schema": <JSON Schema>}

Three properties matter for the science:

* No image, audio or video ever enters a request. `input` is a plain string and
  this module has no code path that attaches media.
* `tools` is never passed, so Search grounding, URL context, code execution and
  file search are all off by construction rather than by a flag we set to false.
* `temperature`, `top_p` and `top_k` are never sent. On this surface they are not
  merely discouraged: `GenerationConfigParam` in the installed SDK has no such
  fields at all.

The SDK is imported lazily so the unit suite runs, and the whole contract is
testable, on a machine with no SDK and no credential.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from ..config import FORBIDDEN_SAMPLING_FIELDS, LLMProviderConfig
from ..contracts import ErrorClass, GenerationRequest, ProviderError, ProviderGenerationResult
from ..firewall import redact_secrets
from .base import RecipeProvider

API_SURFACE = "interactions"
SDK_PACKAGE = "google-genai"


def sdk_version() -> str | None:
    """Installed version, or None. Recorded in provenance; never guessed."""
    try:
        import importlib.metadata as metadata
        return metadata.version(SDK_PACKAGE)
    except Exception:
        return None


#: A 429 whose replenishment is further away than this is treated as exhaustion
#: for the purposes of this run: waiting it out inside a bounded retry budget is
#: not realistic, so the run checkpoints and stops instead of spinning.
RATE_LIMIT_RETRY_CEILING_SECONDS = 300.0

#: "Please retry in 18.039133522s." - the Gemini 429 body's replenishment hint.
_RETRY_HINT_PATTERN = re.compile(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)

#: Markers that name a genuinely long window. A per-day metric does not
#: replenish inside a run, whatever hint accompanies it.
_PER_DAY_MARKERS: tuple[str, ...] = ("per day", "perday", "per_day", "daily limit",
                                     "daily quota", "requests per day")


def _retry_delay_seconds(exc: BaseException, message: str) -> float | None:
    """The provider's own replenishment hint, if it gave one."""
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    match = _RETRY_HINT_PATTERN.search(message)
    return float(match.group(1)) if match else None


def _classify(exc: BaseException) -> ProviderError:
    """Map an SDK/transport exception onto the frozen error vocabulary.

    Classification drives retry, so it is deliberately conservative: anything not
    positively recognised as retryable is treated as non-retryable.
    """
    message = redact_secrets(f"{type(exc).__name__}: {exc}")
    lowered = message.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        status = None

    # Quota exhaustion and rate limiting are both 429 and must not be confused:
    # one backs off and retries, the other stops the run cleanly.
    #
    # C2 finding, observed twice on the live Free Tier: Gemini uses the SAME
    # "You exceeded your current quota" wording for a short-window request limit
    # as for daily exhaustion, and separates them only by the replenishment hint
    # in the body ("Please retry in 18.039133522s."). Matching on the wording
    # alone stopped a pilot that a bounded backoff would have completed, and
    # labelled a transient limit as daily exhaustion in the block artifact.
    # The hint, not the prose, decides.
    quota_markers = ("quota_exceeded", "quota exceeded", "daily quota", "resource_exhausted",
                     "exceeded your current quota", "quota_failure")
    rate_markers = ("rate_limit_exceeded", "rate limit", "too many requests")
    looks_quota = any(marker in lowered for marker in quota_markers)
    looks_rate = status == 429 or any(marker in lowered for marker in rate_markers)
    if looks_quota or looks_rate:
        delay = _retry_delay_seconds(exc, message)
        long_window = any(marker in lowered for marker in _PER_DAY_MARKERS)
        replenishes_soon = (delay is not None and not long_window
                            and delay <= RATE_LIMIT_RETRY_CEILING_SECONDS)
        if replenishes_soon:
            return ProviderError(ErrorClass.RATE_LIMIT, message, status_code=status or 429,
                                 retry_after_seconds=delay)
        if looks_quota:
            # No usable hint, or a window too long to wait out: fail closed.
            return ProviderError(ErrorClass.QUOTA_EXHAUSTED, message, status_code=status or 429)
        return ProviderError(ErrorClass.RATE_LIMIT, message, status_code=status or 429,
                             retry_after_seconds=delay)
    if status in (401, 403) or any(m in lowered for m in
                                   ("api key not valid", "unauthenticated", "permission denied",
                                    "invalid authentication", "missing credential")):
        return ProviderError(ErrorClass.AUTH, message, status_code=status)
    if status == 404 or any(m in lowered for m in ("was not found", "is not found", "unknown model",
                                                   "not supported for", "model not found")):
        return ProviderError(ErrorClass.MODEL_UNAVAILABLE, message, status_code=status)
    if status == 400 or "invalid_argument" in lowered or "unsupported" in lowered:
        return ProviderError(ErrorClass.UNSUPPORTED_CONFIG, message, status_code=status)
    if isinstance(status, int) and 500 <= status < 600:
        return ProviderError(ErrorClass.SERVER_ERROR, message, status_code=status)
    if any(m in lowered for m in ("timeout", "timed out", "connection", "temporarily unavailable",
                                  "remote end closed", "ssl", "network", "unavailable")):
        return ProviderError(ErrorClass.TRANSPORT, message, status_code=status)
    return ProviderError(ErrorClass.SERVER_ERROR, message, status_code=status)


def _extract_text(interaction: Any) -> str | None:
    """Pull the model's text out of an interaction without guessing wildly.

    `output_text` is the documented accessor; the fallbacks exist so an SDK
    revision that renames it produces a contract violation we can see rather
    than a crash.
    """
    text = getattr(interaction, "output_text", None)
    if isinstance(text, str) and text:
        return text
    output = getattr(interaction, "output", None)
    if isinstance(output, str) and output:
        return output
    return None


class GeminiRecipeProvider(RecipeProvider):
    """Calls the Gemini Developer API. Constructed but never invoked by the
    ordinary unit suite."""

    name = "gemini"

    def __init__(self, config: LLMProviderConfig, *, client: Any | None = None) -> None:
        if config.provider != "gemini":
            raise ValueError(f"GeminiRecipeProvider requires provider='gemini', got {config.provider!r}")
        self._config = config
        self._client = client          # injected in tests; never a real client there
        self._sdk_version = sdk_version()

    # --- credential ---------------------------------------------------------
    @property
    def api_key_present(self) -> bool:
        """True/False only. The value is never read into a field, returned,
        logged or serialized."""
        return bool(os.environ.get(self._config.api_key_env, "").strip())

    def _api_key(self) -> str:
        key = os.environ.get(self._config.api_key_env, "").strip()
        if not key:
            raise ProviderError(
                ErrorClass.AUTH,
                f"environment variable {self._config.api_key_env} is not set; "
                "export it in your shell or secret store (never commit it)")
        return key

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError(
                ErrorClass.LOCAL_ERROR,
                f"the {SDK_PACKAGE} package is not installed: {exc}") from exc
        # The key is passed explicitly rather than left to ambient discovery:
        # the SDK also honours GOOGLE_API_KEY and lets it take precedence, which
        # could silently send a different project's credential.
        self._client = genai.Client(api_key=self._api_key())
        return self._client

    # --- request shape ------------------------------------------------------
    def build_call_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """The exact keyword arguments passed to `client.interactions.create`.

        Exposed so a test can assert the wire shape without a network call.
        """
        generation_config: dict[str, Any] = {"thinking_level": self._config.thinking_level}
        if self._config.max_output_tokens is not None:
            generation_config["max_output_tokens"] = int(self._config.max_output_tokens)

        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "input": request.input_text,                    # str: text only, always
            "system_instruction": request.system_instruction,
            "response_format": {
                "type": "text",
                "mime_type": self._config.response_mime_type,
                "schema": request.response_json_schema,
            },
            "generation_config": generation_config,
            "stream": False,
            "store": False,          # no server-side conversation state retained
        }
        # `tools` is deliberately absent: omitting it disables function calling,
        # Search grounding, URL context, code execution and file search together.
        assert "tools" not in kwargs
        assert not any(field in generation_config for field in FORBIDDEN_SAMPLING_FIELDS)
        return kwargs

    # --- generation ---------------------------------------------------------
    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        started = time.monotonic()

        def result(**overrides: Any) -> ProviderGenerationResult:
            base: dict[str, Any] = {
                "slot_id": request.slot_id, "attempt": attempt, "provider": self.name,
                "model_id": request.model_id, "raw_text": None, "parsed": None,
                "latency_seconds": round(time.monotonic() - started, 6),
                "sdk_version": self._sdk_version, "api_surface": API_SURFACE,
            }
            base.update(overrides)
            return ProviderGenerationResult(**base)

        try:
            client = self._ensure_client()
        except ProviderError as exc:
            return result(error=exc)

        try:
            interaction = client.interactions.create(
                **self.build_call_kwargs(request),
                timeout=self._config.request_timeout_seconds,
            )
        except ProviderError as exc:
            return result(error=exc)
        except BaseException as exc:  # noqa: BLE001 - every failure is classified, none escapes
            return result(error=_classify(exc))

        raw_text = _extract_text(interaction)
        if raw_text is None:
            return result(error=ProviderError(
                ErrorClass.CONTRACT_VIOLATION,
                "the interaction carried no text output; the structured-output contract was not honoured"))

        usage = getattr(interaction, "usage", None)
        return result(
            raw_text=raw_text,
            # C2 finding: an Interaction carries `status`, not `finish_reason`.
            # Reading only the documented name recorded a null for every call.
            finish_reason=_first_str(interaction, "finish_reason", "status"),
            usage=_usage_dict(usage),
            provider_request_id=_first_str(interaction, "id"),
            # C2 finding: the resolved model is reported as `model`. It is the
            # only revision the surface exposes, so it is the model_revision.
            model_version=_first_str(interaction, "model_version", "model"),
            provider_seed=None,   # not exposed on this surface; recorded as null
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model_id": self._config.model_id,
                "api_surface": API_SURFACE, "sdk_package": SDK_PACKAGE,
                "sdk_version": self._sdk_version,
                "thinking_level": self._config.thinking_level,
                "response_mime_type": self._config.response_mime_type,
                "api_key_env": self._config.api_key_env,
                "api_key_present": self.api_key_present,
                "tools_passed": False, "media_input_supported": False,
                "forbidden_sampling_fields_sent": []}


def _first_str(source: Any, *names: str) -> str | None:
    """First non-empty attribute among `names`, as a string.

    The SDK types some of these as a Literal union with an `UnrecognizedStr`
    fallback, so the value is stringified rather than compared to a fixed set.
    """
    for name in names:
        value = getattr(source, name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


#: Token counters the installed SDK's `Usage` model actually exposes, verified
#: against google-genai 2.17.0. The names in the older guide (`input_tokens`,
#: `prompt_token_count`, ...) are kept so a rename does not silently drop usage.
_USAGE_SCALAR_FIELDS: tuple[str, ...] = (
    "total_input_tokens", "total_output_tokens", "total_thought_tokens",
    "total_cached_tokens", "total_tool_use_tokens", "total_tokens", "grounding_tool_count",
    "input_tokens", "output_tokens", "thinking_tokens",
    "prompt_token_count", "candidates_token_count", "total_token_count",
)

#: Per-modality breakdowns. Recorded verbatim when present; never synthesised.
_USAGE_MODALITY_FIELDS: tuple[str, ...] = (
    "input_tokens_by_modality", "output_tokens_by_modality",
    "cached_tokens_by_modality", "tool_use_tokens_by_modality",
)


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Whatever the surface reported, and nothing it did not.

    C2 finding: reading only the documented `input_tokens`/`output_tokens` names
    captured `total_tokens` alone from this SDK, so the pilot could not report an
    input/output split that the provider was in fact returning.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {str(key): value for key, value in usage.items() if value is not None}
    collected: dict[str, Any] = {}
    for name in _USAGE_SCALAR_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            collected[name] = value
    for name in _USAGE_MODALITY_FIELDS:
        value = getattr(usage, name, None)
        if value:
            collected[name] = _plain(value)
    return collected


def _plain(value: Any) -> Any:
    """JSON-safe form of an SDK sub-object, without importing its type."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump(mode="json", exclude_none=True))
        except Exception:
            return str(value)
    return str(value)
