"""Provider-neutral request/response contracts for the C1 recipe planner.

No vendor type appears here. `GeminiRecipeProvider` translates into and out of
these structures, so the rest of Version C never imports a vendor SDK.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ErrorClass(str, Enum):
    """Every provider failure lands in exactly one class, and the class alone
    decides whether a retry is allowed. Nothing retries on a guess."""

    # --- retryable ---------------------------------------------------------
    TRANSPORT = "transport"                    # connection reset, DNS, timeout
    SERVER_ERROR = "server_error"              # provider 5xx
    RATE_LIMIT = "rate_limit"                  # 429 rate_limit_exceeded, backoff applies
    INVALID_CANDIDATE = "invalid_candidate"    # transport fine, output failed validation

    # --- non-retryable -----------------------------------------------------
    AUTH = "auth"                              # missing/invalid credential
    MODEL_UNAVAILABLE = "model_unavailable"    # unknown or unsupported model id
    UNSUPPORTED_CONFIG = "unsupported_config"  # provider rejects the frozen config
    QUOTA_EXHAUSTED = "quota_exhausted"        # 429 quota_exceeded / daily quota
    FORBIDDEN_REQUEST = "forbidden_request"    # target-leakage firewall refused it
    CONTRACT_VIOLATION = "contract_violation"  # provider ignored the structured contract
    LOCAL_ERROR = "local_error"                # our own configuration/schema is wrong


RETRYABLE_ERROR_CLASSES: frozenset[ErrorClass] = frozenset({
    ErrorClass.TRANSPORT,
    ErrorClass.SERVER_ERROR,
    ErrorClass.RATE_LIMIT,
    ErrorClass.INVALID_CANDIDATE,
})

NON_RETRYABLE_ERROR_CLASSES: frozenset[ErrorClass] = frozenset({
    ErrorClass.AUTH,
    ErrorClass.MODEL_UNAVAILABLE,
    ErrorClass.UNSUPPORTED_CONFIG,
    ErrorClass.QUOTA_EXHAUSTED,
    ErrorClass.FORBIDDEN_REQUEST,
    ErrorClass.CONTRACT_VIOLATION,
    ErrorClass.LOCAL_ERROR,
})

assert RETRYABLE_ERROR_CLASSES.isdisjoint(NON_RETRYABLE_ERROR_CLASSES)
assert RETRYABLE_ERROR_CLASSES | NON_RETRYABLE_ERROR_CLASSES == set(ErrorClass)


class ProviderError(RuntimeError):
    """A classified provider failure.

    The message must never carry a credential: providers construct these through
    `prism_fas.llm.firewall.redact_secrets`.
    """

    def __init__(self, error_class: ErrorClass, message: str, *,
                 status_code: int | None = None, retry_after_seconds: float | None = None) -> None:
        self.error_class = ErrorClass(error_class)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        return self.error_class in RETRYABLE_ERROR_CLASSES

    def as_dict(self) -> dict[str, Any]:
        return {"error_class": self.error_class.value, "message": str(self),
                "status_code": self.status_code, "retry_after_seconds": self.retry_after_seconds,
                "retryable": self.retryable}


def _stable_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenerationRequest:
    """One request slot. Immutable, hashable and target-free by construction.

    `slot_id` is the stable name of a scientific request slot; a resume must
    regenerate only missing slot ids and never re-run a completed one.
    """

    slot_id: str
    system_instruction: str
    input_text: str
    response_json_schema: dict[str, Any]
    model_id: str
    thinking_level: str
    response_mime_type: str
    max_output_tokens: int | None = None
    recipes_requested: int = 1
    # Identity inputs recorded so a changed prompt/schema/ontology/model cannot
    # silently reuse a previous slot's provenance.
    ontology_identity: str = ""
    prompt_template_identity: str = ""
    provider_config_identity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def identity_material(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "system_instruction_sha256": text_sha256(self.system_instruction),
            "input_text_sha256": text_sha256(self.input_text),
            "response_json_schema_sha256": text_sha256(_stable_text(self.response_json_schema)),
            "model_id": self.model_id,
            "thinking_level": self.thinking_level,
            "response_mime_type": self.response_mime_type,
            "max_output_tokens": self.max_output_tokens,
            "recipes_requested": self.recipes_requested,
            "ontology_identity": self.ontology_identity,
            "prompt_template_identity": self.prompt_template_identity,
            "provider_config_identity": self.provider_config_identity,
        }

    @property
    def request_sha256(self) -> str:
        """Changing the model, prompt, schema, ontology, thinking level or the
        provider config changes this value, and therefore invalidates any
        downstream scientific generation identity bound to it."""
        return text_sha256(_stable_text(self.identity_material()))

    def as_provenance(self) -> dict[str, Any]:
        return {**self.identity_material(), "request_sha256": self.request_sha256}


@dataclass(frozen=True)
class ProviderGenerationResult:
    """One provider attempt, successful or not.

    `raw_text` is preserved verbatim so C2/C3 can archive it and replay it later
    without calling the provider again. A hosted model need not reproduce the
    same text twice, so the archived bytes are the reproducible artifact.
    """

    slot_id: str
    attempt: int
    provider: str
    model_id: str
    raw_text: str | None
    parsed: Any | None
    finish_reason: str | None = None
    latency_seconds: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    provider_request_id: str | None = None
    model_version: str | None = None
    provider_seed: int | None = None
    error: ProviderError | None = None
    sdk_version: str | None = None
    api_surface: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_text is not None

    @property
    def raw_response_sha256(self) -> str | None:
        return text_sha256(self.raw_text) if self.raw_text is not None else None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("parsed", None)
        payload["error"] = self.error.as_dict() if self.error is not None else None
        payload["raw_response_sha256"] = self.raw_response_sha256
        payload["ok"] = self.ok
        # The raw text itself is archived separately by the caller; it is not
        # inlined into every provenance record.
        payload.pop("raw_text", None)
        return payload
