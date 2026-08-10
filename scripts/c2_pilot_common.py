"""Shared plumbing for the C2 live pilot scripts.

Everything the live scripts need to build a frozen request, archive an attempt
verbatim and write a provenance record. The frozen scientific contract is loaded
from the committed C1 config and ontology; nothing here may change it.

The API key is read only by the SDK client inside the provider. It is never
returned, stored, logged or written into any artifact produced from this module.
"""
from __future__ import annotations

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
from prism_fas.llm.json_schema import candidate_json_schema, json_schema_identity  # noqa: E402
from prism_fas.llm.prompt import PromptTemplate, build_generation_prompt, load_prompt_template  # noqa: E402
from prism_fas.llm.providers.base import RecipeProvider  # noqa: E402
from prism_fas.recipes.ontology import Ontology, load_ontology  # noqa: E402

ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
REPORTS = REPO / "reports" / "c2"
RAW_DIR = REPORTS / "raw_responses"

#: The frozen contract this pilot must run under, restated so a drift is loud.
FROZEN = {
    "provider": "gemini",
    "model_id": "gemini-3.6-flash",
    "api_surface": "interactions",
    "thinking_level": "medium",
    "prompt_template_identity": "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f",
    "schema_identity_12x32": "7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4",
    "ontology_identity": "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd",
    "allow_ontology_aliases": False,
}

#: One recipe per pilot slot: the C2 unit of work is a slot, retries stay inside
#: it, and a per-slot batch of one keeps a rejection attributable to one recipe.
RECIPES_PER_PILOT_SLOT = 1


class FrozenContractError(RuntimeError):
    """The loaded contract does not match the frozen C1/C2 values."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class FrozenContext:
    """The loaded, verified frozen contract plus the identities it implies."""

    def __init__(self) -> None:
        self.ontology: Ontology = load_ontology(ONTOLOGY_PATH)
        self.config: LLMProviderConfig = load_llm_config(LLM_CONFIG_PATH)
        self.template: PromptTemplate = load_prompt_template(self.ontology)
        self.provider_config_identity = provider_config_identity(self.config)
        self.schema_identity_12x32 = json_schema_identity(
            candidate_json_schema(self.ontology, recipes_requested=32))
        self.verify()

    def verify(self) -> None:
        actual = {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "api_surface": self.config.api_surface,
            "thinking_level": self.config.thinking_level,
            "prompt_template_identity": self.template.identity(),
            "schema_identity_12x32": self.schema_identity_12x32,
            "ontology_identity": self.ontology.sha256,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,
        }
        drift = {key: {"frozen": FROZEN[key], "actual": actual[key]}
                 for key in FROZEN if FROZEN[key] != actual[key]}
        if drift:
            raise FrozenContractError(f"the frozen C2 contract has drifted: {drift}")

    def schema(self, recipes_requested: int) -> dict[str, Any]:
        return candidate_json_schema(self.ontology, recipes_requested=recipes_requested)

    def request(self, slot_id: str, *, recipes_requested: int = RECIPES_PER_PILOT_SLOT,
                metadata: dict[str, Any] | None = None) -> GenerationRequest:
        """One frozen, target-free request. Text only; no media path exists."""
        schema = self.schema(recipes_requested)
        return GenerationRequest(
            slot_id=slot_id,
            system_instruction=self.template.system_instruction,
            input_text=build_generation_prompt(self.template, recipes_requested=recipes_requested),
            response_json_schema=schema,
            model_id=self.config.model_id,
            thinking_level=self.config.thinking_level,
            response_mime_type=self.config.response_mime_type,
            max_output_tokens=self.config.max_output_tokens,
            recipes_requested=recipes_requested,
            ontology_identity=self.ontology.sha256,
            prompt_template_identity=self.template.identity(),
            provider_config_identity=self.provider_config_identity,
            metadata=dict(metadata or {}),
        )

    def config_summary(self) -> dict[str, Any]:
        summary = self.config.public_summary()
        summary["recipe_schema_version"] = self.ontology.recipe_schema_version
        return summary

    def as_frozen_record(self) -> dict[str, Any]:
        return {
            "provider": "Google Gemini Developer API",
            "provider_name": self.config.provider,
            "model_id": self.config.model_id,
            "api_surface": self.config.api_surface,
            "thinking_level": self.config.thinking_level,
            "response_mime_type": self.config.response_mime_type,
            "max_output_tokens": self.config.max_output_tokens,
            "prompt_template_identity": self.template.identity(),
            "system_prompt_sha256": self.template.system_instruction_sha256,
            "request_template_sha256": self.template.generation_template_sha256,
            "ontology_identity": self.ontology.sha256,
            "schema_identity_12x32_reference": self.schema_identity_12x32,
            "schema_identity_per_pilot_slot": json_schema_identity(
                self.schema(RECIPES_PER_PILOT_SLOT)),
            "provider_config_identity": self.provider_config_identity,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,
            "recipes_per_pilot_slot": RECIPES_PER_PILOT_SLOT,
            "schema_identity_note":
                "The frozen 12x32 schema identity is the reference recorded at C1 for the C3 "
                "schedule. A pilot slot asks for one recipe, so its schema instance carries "
                "minItems/maxItems = 1 and therefore a different SHA-256. The schema BUILDER, "
                "the ontology it is built from and every enum, range and rule inside it are "
                "unchanged; only the batch size differs.",
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
    """Delegates to a real provider and archives every attempt verbatim.

    One attempt in, one archive record out, whether it succeeded or failed. This
    is what makes "every attempt has provenance" structural rather than a habit.
    """

    def __init__(self, inner: RecipeProvider, *, phase: str) -> None:
        self.name = inner.name
        self._inner = inner
        self._phase = phase
        self.records: list[dict[str, Any]] = []

    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        result = self._inner.generate(request, attempt=attempt)
        self.records.append(archive_record(request, result, phase=self._phase))
        return result

    def describe(self) -> dict[str, Any]:
        return {**self._inner.describe(), "recording": True, "phase": self._phase}


def archive_record(request: GenerationRequest, result: ProviderGenerationResult, *,
                   phase: str) -> dict[str, Any]:
    """One archived attempt. `raw_text` is the verbatim provider output."""
    return {
        "phase": phase,
        "slot_id": result.slot_id,
        "attempt": int(result.attempt),
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
        "raw_text": result.raw_text,
        "raw_response_sha256": result.raw_response_sha256,
        "latency_seconds": result.latency_seconds,
        "recorded_at_utc": utc_now(),
        "error": result.error.as_dict() if result.error is not None else None,
    }


def replay_fields(record: dict[str, Any]) -> dict[str, Any]:
    """The subset `ReplayArchive.from_records` accepts."""
    return {key: record[key] for key in
            ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
             "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
             "api_surface", "request_sha256")}


def write_raw_response_files(records: list[dict[str, Any]], subdir: str) -> list[str]:
    """Verbatim raw text on disk, one file per attempt.

    `reports/c2/**/raw_responses/` is git-ignored by repository policy: the raw
    archive is provenance held on disk and referenced by hash. The same bytes are
    also carried in the committed `C2_RAW_RESPONSE_ARCHIVE.json` so replay works
    from a clean checkout.
    """
    target = RAW_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for record in records:
        # `sequence` is the per-slot call counter. The frozen retry policy resets
        # `attempt` after a transport failure, so it does not identify a call.
        marker = record.get("sequence")
        name = (f"{record['slot_id']}__seq{marker:02d}.txt" if marker is not None
                else f"{record['slot_id']}__attempt{record['attempt']:02d}.txt")
        path = target / name
        path.write_text(record["raw_text"] if record["raw_text"] is not None else "",
                        encoding="utf-8")
        paths.append(repo_relative(path))
    return paths


def repo_relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", repo_relative(path))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def api_key_present(config: LLMProviderConfig) -> bool:
    """True/False only. The value is never read into a variable that is returned."""
    return bool(os.environ.get(config.api_key_env, "").strip())


__all__ = ["REPO", "REPORTS", "RAW_DIR", "FROZEN", "RECIPES_PER_PILOT_SLOT", "FrozenContext",
           "FrozenContractError", "RecordingProvider", "archive_record", "replay_fields",
           "write_raw_response_files", "write_json", "read_json", "api_key_present", "git",
           "utc_now"]
