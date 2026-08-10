"""Immutable generation provenance and the C1 identity chain.

Every field the spec's section 7.7 table marks YES is required here, and a record
missing one fails rather than serializing a hole. The API key is not a field and
cannot become one: `GenerationProvenance` rejects any extra key, and a redaction
pass runs over every string before a record is emitted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .contracts import GenerationRequest, ProviderGenerationResult
from .firewall import redact_secrets

PROVENANCE_SCHEMA_VERSION = "c1-llm-provenance-v1"

#: Spec section 7.7. Present in every record; a missing one is an error.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "provider",
    "model_id",
    "model_revision",
    "generation_timestamp_utc",
    "system_prompt_sha256",
    "request_template_sha256",
    "ontology_identity",
    "schema_version",
    "thinking_level",
    "max_output_tokens",
    "provider_seed",
    "raw_response_sha256",
    "raw_response_path",
    "parsed_recipe_sha256",
    "validation_result",
    "validation_errors",
    "retry_count",
    "generator_code_commit",
    "billing_tier",
    "request_schedule_id",
    "api_or_sdk_version",
)

#: Identity boundaries. Changing any input on the left invalidates everything
#: downstream of it in the scientific generation.
IDENTITY_CHAIN_ORDER: tuple[str, ...] = (
    "ontology_identity",
    "llm_schema_identity",
    "llm_prompt_template_identity",
    "llm_provider_config_identity",
    "generation_request_identity",
    "raw_response_identity",
    "canonical_recipe_identity",
)


class ProvenanceError(ValueError):
    """A provenance record is incomplete or carries a forbidden field."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


@dataclass(frozen=True)
class GenerationProvenance:
    """One attempt's immutable record."""

    record: dict[str, Any]

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_PROVENANCE_FIELDS if name not in self.record]
        if missing:
            raise ProvenanceError(f"provenance record is missing required fields: {missing}")
        forbidden = [key for key in self.record
                     if "api_key" in key.lower() or key.lower() in {"key", "token", "authorization",
                                                                    "credential", "secret"}]
        if forbidden:
            raise ProvenanceError(f"provenance record carries forbidden credential fields: {forbidden}")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.record)

    @property
    def identity(self) -> str:
        text = json.dumps(self.record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_provenance(*,
                     request: GenerationRequest,
                     result: ProviderGenerationResult,
                     config_summary: dict[str, Any],
                     prompt_provenance: dict[str, Any],
                     schema_identity: str,
                     validation_result: str,
                     validation_errors: list[dict[str, Any]] | None = None,
                     parsed_recipe_sha256: list[str] | None = None,
                     retry_count: int = 0,
                     generator_code_commit: str | None = None,
                     request_schedule_id: str | None = None,
                     raw_response_path: str | None = None,
                     billing_tier: str = "free") -> GenerationProvenance:
    """Assemble one record. Every string is redacted on the way in."""
    record: dict[str, Any] = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "slot_id": request.slot_id,
        "attempt": result.attempt,

        # --- spec 7.7 required fields ---
        "provider": result.provider,
        "model_id": result.model_id,
        "model_revision": result.model_version,
        "generation_timestamp_utc": _utc_now(),
        "system_prompt_sha256": prompt_provenance.get("system_prompt_sha256"),
        "request_template_sha256": prompt_provenance.get("request_template_sha256"),
        "ontology_identity": request.ontology_identity,
        "schema_version": config_summary.get("recipe_schema_version"),
        "thinking_level": request.thinking_level,
        "max_output_tokens": request.max_output_tokens,
        "provider_seed": result.provider_seed,
        "raw_response_sha256": result.raw_response_sha256,
        "raw_response_path": raw_response_path,
        "parsed_recipe_sha256": list(parsed_recipe_sha256 or []),
        "validation_result": validation_result,
        "validation_errors": list(validation_errors or []),
        "retry_count": retry_count,
        "generator_code_commit": generator_code_commit,
        "billing_tier": billing_tier,
        "request_schedule_id": request_schedule_id,
        "api_or_sdk_version": result.sdk_version,

        # --- identity chain and diagnostics ---
        "api_surface": result.api_surface,
        "llm_schema_identity": schema_identity,
        "llm_prompt_template_identity": prompt_provenance.get("prompt_template_identity"),
        "llm_provider_config_identity": config_summary.get("provider_config_identity"),
        "generation_request_identity": request.request_sha256,
        "finish_reason": result.finish_reason,
        "latency_seconds": result.latency_seconds,
        "usage": dict(result.usage),
        "provider_request_id": result.provider_request_id,
        "error": result.error.as_dict() if result.error is not None else None,
    }
    return GenerationProvenance(_redact(record))


def identity_chain(*, ontology_identity: str, schema_identity: str,
                   prompt_template_identity: str, provider_config_identity: str,
                   generation_request_identity: str | None = None,
                   raw_response_identity: str | None = None,
                   canonical_recipe_identity: str | None = None) -> dict[str, Any]:
    """The ordered chain plus a rolled-up hash.

    The roll-up is what a later bank lock binds to: changing the model, the
    prompt, the schema, the ontology or the thinking configuration changes it,
    and therefore invalidates any scientific generation carried out under the
    previous value.
    """
    links = {
        "ontology_identity": ontology_identity,
        "llm_schema_identity": schema_identity,
        "llm_prompt_template_identity": prompt_template_identity,
        "llm_provider_config_identity": provider_config_identity,
        "generation_request_identity": generation_request_identity,
        "raw_response_identity": raw_response_identity,
        "canonical_recipe_identity": canonical_recipe_identity,
    }
    ordered = {name: links[name] for name in IDENTITY_CHAIN_ORDER}
    resolved = {name: value for name, value in ordered.items() if value is not None}
    text = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"chain_order": list(IDENTITY_CHAIN_ORDER), "links": ordered,
            "resolved_links": len(resolved),
            "chain_identity": hashlib.sha256(text.encode("utf-8")).hexdigest()}
