"""Typed, fail-closed configuration for the C1 recipe planner.

Unknown keys are rejected rather than ignored, so a typo cannot silently change
scientific behaviour. Every field that could alter what the model produces enters
`provider_config_identity`; operational fields (timeouts, backoff seconds, the
env var name) are excluded from that identity and are documented as such.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Verified against the installed SDK's ThinkingLevel literal and the official
# thinking guide (see reports/c1/C1_PROVIDER_AUDIT.json).
THINKING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")
SUPPORTED_PROVIDERS: tuple[str, ...] = ("gemini", "mock", "replay")
SUPPORTED_API_SURFACES: tuple[str, ...] = ("interactions",)
SUPPORTED_RESPONSE_MIME_TYPES: tuple[str, ...] = ("application/json",)

# Sampling controls that must never be sent for Gemini 3.x. They are also
# structurally absent from GenerationConfigParam in the installed SDK, so this
# list exists to make the prohibition explicit and testable, not to filter.
FORBIDDEN_SAMPLING_FIELDS: tuple[str, ...] = ("temperature", "top_p", "top_k")


class LLMConfigError(ValueError):
    """The LLM configuration is invalid. Never partially applied."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RetryConfig(_Strict):
    """Bounded. `semantic_max_retries` is the spec's 'retry maximum = 2 per
    request'; transport attempts are counted separately so a flaky connection
    cannot consume the semantic budget."""

    semantic_max_retries: int = Field(default=2, ge=0, le=5)
    transport_max_attempts: int = Field(default=4, ge=1, le=10)
    backoff_initial_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    backoff_max_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)

    @model_validator(mode="after")
    def _bounded(self) -> "RetryConfig":
        if self.backoff_initial_seconds > self.backoff_max_seconds:
            raise ValueError("backoff_initial_seconds cannot exceed backoff_max_seconds")
        return self


class QuotaConfig(_Strict):
    """Free Tier first. Code never enables billing; `billing_tier` records what
    the user has already enabled and is provenance, not a treatment factor."""

    billing_tier: Literal["free", "paid"] = "free"
    auto_enable_paid: Literal[False] = False
    on_quota_exhausted: Literal["checkpoint_and_stop"] = "checkpoint_and_stop"
    quota_block_filename: str = "API_QUOTA_BLOCKED.json"

    @field_validator("quota_block_filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("quota_block_filename must be a bare filename")
        return value


class LLMProviderConfig(_Strict):
    """The frozen provider contract.

    `api_key_env` names an environment variable. A real key is never stored here,
    never defaulted, and never serialized.
    """

    config_schema_version: str = "c1-llm-provider-config-v1"
    provider: str
    model_id: str
    api_surface: str = "interactions"
    sdk_package: str = "google-genai"
    sdk_version_pin: str | None = None

    thinking_level: str = "medium"
    response_mime_type: str = "application/json"
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)

    # Every capability that could turn a text-only prior into retrieval,
    # execution or visual adaptation. All must stay false.
    tools_enabled: Literal[False] = False
    grounding_enabled: Literal[False] = False
    url_context_enabled: Literal[False] = False
    code_execution_enabled: Literal[False] = False
    file_search_enabled: Literal[False] = False
    image_input_enabled: Literal[False] = False
    audio_input_enabled: Literal[False] = False
    video_input_enabled: Literal[False] = False
    store_interaction: Literal[False] = False

    # Strictest reading of spec 7.6 ("unknown enum -> REJECT"): the ontology's
    # documented alias map is OFF for the scientific path, so a non-canonical
    # spelling is a rejected candidate rather than a silently normalized one.
    # It is identity-bearing: enabling it changes the recipe bank identity.
    allow_ontology_aliases: bool = False

    api_key_env: str = "GEMINI_API_KEY"
    request_timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)

    retry: RetryConfig = Field(default_factory=RetryConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported provider {value!r}; allowed {list(SUPPORTED_PROVIDERS)}")
        return value

    @field_validator("api_surface")
    @classmethod
    def _api_surface(cls, value: str) -> str:
        if value not in SUPPORTED_API_SURFACES:
            raise ValueError(f"unsupported api_surface {value!r}; allowed {list(SUPPORTED_API_SURFACES)}")
        return value

    @field_validator("thinking_level")
    @classmethod
    def _thinking(cls, value: str) -> str:
        if value not in THINKING_LEVELS:
            raise ValueError(f"unsupported thinking_level {value!r}; allowed {list(THINKING_LEVELS)}")
        return value

    @field_validator("response_mime_type")
    @classmethod
    def _mime(cls, value: str) -> str:
        if value not in SUPPORTED_RESPONSE_MIME_TYPES:
            raise ValueError(f"unsupported response_mime_type {value!r}; "
                             f"allowed {list(SUPPORTED_RESPONSE_MIME_TYPES)}")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env(cls, value: str) -> str:
        if not value or not value.replace("_", "").isalnum() or not value.isupper():
            raise ValueError("api_key_env must be an UPPER_SNAKE_CASE environment variable name")
        # A value that looks like a credential means someone pasted the key here.
        if value.startswith("AIza") or len(value) > 64:
            raise ValueError("api_key_env must name an environment variable, not carry a key")
        return value

    @model_validator(mode="after")
    def _scientific_model(self) -> "LLMProviderConfig":
        if self.provider == "gemini" and not self.model_id:
            raise ValueError("the gemini provider requires an explicit model_id")
        return self

    # --- identity -----------------------------------------------------------
    def identity_material(self) -> dict[str, Any]:
        """Only fields that can change what the model produces. Timeouts, backoff
        and the env var name are operational and stay out."""
        return {
            "config_schema_version": self.config_schema_version,
            "provider": self.provider,
            "model_id": self.model_id,
            "api_surface": self.api_surface,
            "sdk_package": self.sdk_package,
            "thinking_level": self.thinking_level,
            "response_mime_type": self.response_mime_type,
            "max_output_tokens": self.max_output_tokens,
            "tools_enabled": self.tools_enabled,
            "grounding_enabled": self.grounding_enabled,
            "url_context_enabled": self.url_context_enabled,
            "code_execution_enabled": self.code_execution_enabled,
            "file_search_enabled": self.file_search_enabled,
            "image_input_enabled": self.image_input_enabled,
            "audio_input_enabled": self.audio_input_enabled,
            "video_input_enabled": self.video_input_enabled,
            "store_interaction": self.store_interaction,
            "allow_ontology_aliases": self.allow_ontology_aliases,
            "semantic_max_retries": self.retry.semantic_max_retries,
        }

    def public_summary(self) -> dict[str, Any]:
        """Safe to write into a report: names the env var, never reads it."""
        return {**self.identity_material(),
                "api_key_env": self.api_key_env,
                "sdk_version_pin": self.sdk_version_pin,
                "billing_tier": self.quota.billing_tier,
                "auto_enable_paid": self.quota.auto_enable_paid,
                "on_quota_exhausted": self.quota.on_quota_exhausted,
                "forbidden_sampling_fields": list(FORBIDDEN_SAMPLING_FIELDS),
                "provider_config_identity": provider_config_identity(self)}


def provider_config_identity(config: LLMProviderConfig) -> str:
    text = json.dumps(config.identity_material(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_llm_config(payload: Any) -> LLMProviderConfig:
    if not isinstance(payload, dict):
        raise LLMConfigError("llm config must be a mapping")
    try:
        return LLMProviderConfig.model_validate(payload)
    except ValidationError as exc:
        raise LLMConfigError(str(exc)) from exc


def load_llm_config(path: Path) -> LLMProviderConfig:
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LLMConfigError(f"llm config not found: {path}") from exc
    return parse_llm_config(payload)
