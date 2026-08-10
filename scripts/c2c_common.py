"""Shared plumbing for C2C: the scientific route-contract repair.

C2C changes exactly two things relative to the accepted C2B contract:

* the system instruction gains a mandatory route-declaration rule, generated
  from the route policy so the prompt and the validator cannot drift;
* the validation pipeline enforces `SCIENTIFIC_ROUTE_POLICY` before
  canonicalization, duplicate detection and the compiler.

Everything else is inherited unchanged: ontology, item schema, batch envelope,
coverage quotas, provider, model, thinking level, alias policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.llm.config import LLMProviderConfig, load_llm_config, provider_config_identity  # noqa: E402
from prism_fas.llm.contracts import GenerationRequest, ProviderGenerationResult  # noqa: E402
from prism_fas.llm.coverage_quotas import QuotaSpec, load_quota_spec  # noqa: E402
from prism_fas.llm.json_schema import (candidate_json_schema, candidate_object_schema,  # noqa: E402
                                       json_schema_identity)
from prism_fas.llm.prompt import (PromptTemplate, build_generation_prompt,  # noqa: E402
                                  load_prompt_template)
from prism_fas.llm.providers.base import RecipeProvider  # noqa: E402
from prism_fas.llm.route_policy import RoutePolicy, load_route_policy  # noqa: E402
from prism_fas.recipes.ontology import Ontology, load_ontology  # noqa: E402

ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
QUOTA_PATH = REPO / "configs" / "version_c" / "llm" / "c2b_coverage_quotas.yaml"
ROUTE_POLICY_PATH = REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml"
REPORTS = REPO / "reports" / "c2c"
DOCS = REPO / "docs" / "c2c"
RAW_DIR = REPORTS / "raw_responses"
C2B_REPORTS = REPO / "reports" / "c2b"

LOGICAL_BATCH_ID = "C2C_BATCH_000"
BATCH_SIZE = 32
C2C_BANK_ID = "c2c-batch-disposable"
C2B_BANK_ID = "c2b-batch-disposable"
BATCH_REQUEST_TIMEOUT_SECONDS = 600.0
ARRAY_BOUNDS_SENT = False

#: C3 schedule, recorded but NOT executed.
C3_REQUESTS = 12
C3_RAW_SLOTS = C3_REQUESTS * BATCH_SIZE
C3_MIN_UNIQUE_POOL = 320
C3_FINAL_BANK = 256

#: Inherited from the accepted C2B contract; C2C must not move any of these.
FROZEN = {
    "provider": "gemini",
    "model_id": "gemini-3.6-flash",
    "api_surface": "interactions",
    "thinking_level": "medium",
    "ontology_identity": "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd",
    "single_recipe_schema_identity":
        "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579",
    "batch_envelope_schema_identity":
        "f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113",
    "coverage_quota_identity":
        "89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5",
    "allow_ontology_aliases": False,
    "provider_config_identity":
        "3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da",
}

#: The prompt identity C2B ran under, kept so the amendment is auditable.
C2B_SYSTEM_PROMPT_IDENTITY = "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f"

#: C2B coverage, for the "did the route repair damage coverage?" comparison.
C2B_COVERAGE = {
    "media": {"present": 5, "total": 5, "max_share_percent": 21.875},
    "geometry": {"present": 6, "total": 6, "max_share_percent": 18.75},
    "illumination": {"present": 6, "total": 6, "max_share_percent": 18.75},
    "artifacts": {"present": 8, "total": 8, "max_share_percent": 16.4179},
    "regions": {"present": 9, "total": 9, "max_share_percent": 12.1622},
}


class FrozenContractError(RuntimeError):
    """The loaded contract does not match the accepted C2B freeze."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class RouteContext:
    """The C2C contract: C2B's, plus the frozen scientific route policy."""

    def __init__(self) -> None:
        self.ontology: Ontology = load_ontology(ONTOLOGY_PATH)
        loaded: LLMProviderConfig = load_llm_config(LLM_CONFIG_PATH)
        self.config_timeout_before_override = loaded.request_timeout_seconds
        # Operational only; excluded from `identity_material` by design.
        self.config: LLMProviderConfig = loaded.model_copy(
            update={"request_timeout_seconds": BATCH_REQUEST_TIMEOUT_SECONDS})
        self.quotas: QuotaSpec = load_quota_spec(QUOTA_PATH)
        self.quotas.validate_against(self.ontology)
        self.route_policy: RoutePolicy = load_route_policy(ROUTE_POLICY_PATH)
        self.route_policy.validate_against(self.ontology)

        # The prompt before and after the minimal amendment, so the diff is
        # always reproducible from the code rather than transcribed.
        self.template_before: PromptTemplate = load_prompt_template(self.ontology)
        self.template: PromptTemplate = load_prompt_template(self.ontology, self.route_policy)

        self.provider_config_identity = provider_config_identity(self.config)
        self.item_schema = candidate_object_schema(self.ontology)
        self.batch_schema = candidate_json_schema(
            self.ontology, recipes_requested=BATCH_SIZE, array_bounds=ARRAY_BOUNDS_SENT)
        self.single_recipe_schema_identity = json_schema_identity(self.item_schema)
        self.batch_envelope_schema_identity = json_schema_identity(self.batch_schema)
        self.verify()

    def verify(self) -> None:
        actual = {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "api_surface": self.config.api_surface,
            "thinking_level": self.config.thinking_level,
            "ontology_identity": self.ontology.sha256,
            "single_recipe_schema_identity": self.single_recipe_schema_identity,
            "batch_envelope_schema_identity": self.batch_envelope_schema_identity,
            "coverage_quota_identity": self.quotas.quota_identity,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,
            "provider_config_identity": self.provider_config_identity,
        }
        drift = {key: {"frozen": FROZEN[key], "actual": actual[key]}
                 for key in FROZEN if FROZEN[key] != actual[key]}
        if drift:
            raise FrozenContractError(f"the accepted C2B contract has drifted: {drift}")
        if self.template_before.identity() != C2B_SYSTEM_PROMPT_IDENTITY:
            raise FrozenContractError(
                "the pre-amendment prompt no longer reproduces the C2B identity; the route "
                "block is supposed to be the ONLY prompt change")

    # --- request ------------------------------------------------------------
    def batch_input_text(self) -> str:
        return build_generation_prompt(
            self.template, recipes_requested=BATCH_SIZE,
            coverage_quotas=self.quotas.prompt_block(self.ontology))

    @property
    def batch_template_identity(self) -> str:
        return hashlib.sha256(self.batch_input_text().encode("utf-8")).hexdigest()

    def request(self, batch_id: str = LOGICAL_BATCH_ID) -> GenerationRequest:
        return GenerationRequest(
            slot_id=batch_id,
            system_instruction=self.template.system_instruction,
            input_text=self.batch_input_text(),
            response_json_schema=self.batch_schema,
            model_id=self.config.model_id,
            thinking_level=self.config.thinking_level,
            response_mime_type=self.config.response_mime_type,
            max_output_tokens=self.config.max_output_tokens,
            recipes_requested=BATCH_SIZE,
            ontology_identity=self.ontology.sha256,
            prompt_template_identity=self.template.identity(),
            provider_config_identity=self.provider_config_identity,
            route_policy_identity=self.route_policy.route_policy_identity,
            metadata={"phase": "c2c_route_contract", "disposable": True, "enters_c3": False},
        )

    def config_summary(self) -> dict[str, Any]:
        summary = self.config.public_summary()
        summary["recipe_schema_version"] = self.ontology.recipe_schema_version
        return summary

    def prompt_diff(self) -> dict[str, Any]:
        """The exact byte-level amendment, computed rather than transcribed."""
        import difflib

        before = self.template_before.system_instruction
        after = self.template.system_instruction
        diff = list(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="system_instruction (C1/C2/C2B)", tofile="system_instruction (C2C)", n=2))
        added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
        removed = [line[1:] for line in diff
                   if line.startswith("-") and not line.startswith("---")]
        return {
            "reason": "C2B_VALIDATOR_COMPILER_ROUTE_CONTRACT_REPAIR",
            "classification": "source-independent engineering / spec reconciliation, NOT target "
                              "tuning. No dataset, metric, attack family or target result was "
                              "consulted; the amendment states an execution contract the frozen "
                              "Version-C synthesis design already required.",
            "old_prompt_identity": self.template_before.identity(),
            "new_prompt_identity": self.template.identity(),
            "old_system_prompt_sha256": self.template_before.system_instruction_sha256,
            "new_system_prompt_sha256": self.template.system_instruction_sha256,
            "generation_template_changed": (self.template_before.generation_template
                                            != self.template.generation_template),
            "coverage_quotas_changed": False,
            "chars_before": len(before),
            "chars_after": len(after),
            "chars_added": len(after) - len(before),
            "lines_added": len(added),
            "lines_removed": len(removed),
            "added_lines": added,
            "removed_lines": removed,
            "unified_diff": "".join(diff),
        }

    def as_contract_record(self) -> dict[str, Any]:
        return {
            "provider": "Google Gemini Developer API",
            "provider_name": self.config.provider,
            "model_id": self.config.model_id,
            "api_surface": self.config.api_surface,
            "sdk_package": self.config.sdk_package,
            "thinking_level": self.config.thinking_level,
            "response_mime_type": self.config.response_mime_type,
            "max_output_tokens": self.config.max_output_tokens,

            "system_prompt_identity": self.template.identity(),
            "system_prompt_sha256": self.template.system_instruction_sha256,
            "system_prompt_identity_before_c2c": self.template_before.identity(),
            "batch_generation_template_identity": self.batch_template_identity,
            "coverage_quota_identity": self.quotas.quota_identity,
            "coverage_quotas_changed_in_c2c": False,

            "single_recipe_schema_identity": self.single_recipe_schema_identity,
            "single_recipe_schema_changed_in_c2c": False,
            "batch_envelope_schema_identity": self.batch_envelope_schema_identity,
            "batch_envelope_array_bounds_sent": ARRAY_BOUNDS_SENT,
            "exact_batch_size_enforced_locally": BATCH_SIZE,

            "ontology_identity": self.ontology.sha256,
            "route_policy_identity": self.route_policy.route_policy_identity,
            "route_policy_version": self.route_policy.version,
            "required_generator_route": list(
                self.route_policy.allowed_scientific_generator_route),
            "provider_config_identity": self.provider_config_identity,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,

            "retry_policy": {
                "semantic_max_retries": self.config.retry.semantic_max_retries,
                "transport_max_attempts": self.config.retry.transport_max_attempts,
                "backoff_initial_seconds": self.config.retry.backoff_initial_seconds,
                "backoff_multiplier": self.config.retry.backoff_multiplier,
                "backoff_max_seconds": self.config.retry.backoff_max_seconds,
            },
            "request_schedule_for_c3": (
                f"{C3_REQUESTS} requests x {BATCH_SIZE} recipes = {C3_RAW_SLOTS} raw slots; "
                f"minimum unique pool {C3_MIN_UNIQUE_POOL}; final bank {C3_FINAL_BANK}"),

            "sampling_controls_sent": [],
            "tools_sent": False,
            "grounding": False,
            "url_context": False,
            "file_search": False,
            "code_execution": False,
            "media_inputs": False,
            "input_type": "str (text only)",
            "store_interaction": False,
        }


class RecordingProvider(RecipeProvider):
    """Delegates to a real provider and archives every attempt verbatim."""

    def __init__(self, inner: RecipeProvider, *, phase: str) -> None:
        self.name = inner.name
        self._inner = inner
        self._phase = phase
        self.records: list[dict[str, Any]] = []

    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        result = self._inner.generate(request, attempt=attempt)
        self.records.append(archive_record(request, result, phase=self._phase,
                                           sequence=len(self.records) + 1))
        return result

    def describe(self) -> dict[str, Any]:
        return {**self._inner.describe(), "recording": True, "phase": self._phase}


def archive_record(request: GenerationRequest, result: ProviderGenerationResult, *,
                   phase: str, sequence: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "logical_batch_id": request.slot_id,
        "slot_id": result.slot_id,
        "attempt": int(result.attempt),
        "sequence": sequence,
        "provider": result.provider,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "finish_reason": result.finish_reason,
        "usage": dict(result.usage),
        "provider_request_id": result.provider_request_id,
        "provider_seed": result.provider_seed,
        "sdk_version": result.sdk_version,
        "api_surface": result.api_surface,
        "request_sha256": request.request_sha256,
        "route_policy_identity": request.route_policy_identity,
        "raw_text": result.raw_text,
        "raw_response_sha256": result.raw_response_sha256,
        "latency_seconds": result.latency_seconds,
        "recorded_at_utc": utc_now(),
        "error": result.error.as_dict() if result.error is not None else None,
    }


def replay_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in
            ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
             "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
             "api_surface", "request_sha256")}


def repo_relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_raw_response_files(records: list[dict[str, Any]]) -> list[str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for record in records:
        path = RAW_DIR / f"{record['logical_batch_id']}__seq{record['sequence']:02d}.json"
        path.write_text(record["raw_text"] or "", encoding="utf-8")
        paths.append(repo_relative(path))
    return paths


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", repo_relative(path))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def api_key_present(config: LLMProviderConfig) -> bool:
    return bool(os.environ.get(config.api_key_env, "").strip())


__all__ = ["REPO", "REPORTS", "DOCS", "RAW_DIR", "C2B_REPORTS", "FROZEN", "BATCH_SIZE",
           "LOGICAL_BATCH_ID", "C2C_BANK_ID", "C2B_BANK_ID", "C2B_COVERAGE", "C3_REQUESTS",
           "C3_RAW_SLOTS", "C3_MIN_UNIQUE_POOL", "C3_FINAL_BANK", "RouteContext",
           "FrozenContractError", "RecordingProvider", "archive_record", "replay_fields",
           "write_raw_response_files", "write_json", "read_json", "api_key_present", "git",
           "utc_now", "repo_relative"]
