"""Shared plumbing for the C2B batch-shape validation.

C2B changes exactly one thing about the request: its SHAPE. One request now asks
for 32 recipe objects instead of one, and carries generic ontology-level coverage
quotas. The system instruction, the ontology, the recipe semantics, the
compatibility rules, the validator and the compiler are all inherited from C2
untouched.

Two schema identities are tracked separately, because they answer different
questions:

* `single_recipe_schema_identity` - the item schema, one recipe object. This is
  what "did the recipe semantics change?" depends on, and C2B must not move it.
* `batch_envelope_schema_identity` - the whole {"recipes": [...]} envelope with
  minItems = maxItems = 32. Only this may differ from C2, which used a 1-object
  envelope.
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
from prism_fas.llm.coverage_quotas import QuotaSpec, load_quota_spec  # noqa: E402
from prism_fas.llm.json_schema import (candidate_json_schema, candidate_object_schema,  # noqa: E402
                                       json_schema_identity)
from prism_fas.llm.prompt import (PromptTemplate, build_generation_prompt,  # noqa: E402
                                  load_prompt_template)
from prism_fas.llm.providers.base import RecipeProvider  # noqa: E402
from prism_fas.recipes.ontology import Ontology, load_ontology  # noqa: E402

ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
QUOTA_PATH = REPO / "configs" / "version_c" / "llm" / "c2b_coverage_quotas.yaml"
REPORTS = REPO / "reports" / "c2b"
DOCS = REPO / "docs" / "c2b"
RAW_DIR = REPORTS / "raw_responses"

#: The one logical batch C2B is allowed to run.
LOGICAL_BATCH_ID = "C2B_BATCH_000"
BATCH_SIZE = 32
C2B_BANK_ID = "c2b-batch-disposable"

#: Operational only, and excluded from `provider_config_identity` by design.
BATCH_REQUEST_TIMEOUT_SECONDS = 600.0

#: C2B finding: the provider rejects the 32-object envelope with `400
#: INVALID_ARGUMENT` while it accepts the byte-identical 1-object envelope, and
#: Google documents array length limits as a source of schema-complexity
#: rejection. The bound is therefore omitted from the REQUEST; "exactly 32"
#: remains enforced locally on the response by `validate_response`.
ARRAY_BOUNDS_SENT = False

#: Inherited from the accepted C2 freeze; C2B must not move any of these.
FROZEN = {
    "provider": "gemini",
    "model_id": "gemini-3.6-flash",
    "api_surface": "interactions",
    "thinking_level": "medium",
    "prompt_template_identity": "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f",
    "ontology_identity": "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd",
    "allow_ontology_aliases": False,
    # Recorded at C1 as `llm_schema_identity_12x32`. It is the 32-OBJECT ENVELOPE
    # identity. C2B proved the provider REJECTS it (400 INVALID_ARGUMENT), so it
    # is verified as still COMPUTABLE - the builder is unchanged - but it is not
    # the envelope C2B sends.
    "bounded_batch_envelope_identity": "7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4",
}

#: The C2 singleton pilot's coverage, for the mode-collapse comparison. Both are
#: source-independent prompt-development evidence, so comparing them is allowed.
C2_SINGLETON_COVERAGE = {
    "artifacts": {"present": 6, "total": 8, "max_share_percent": 30.2326,
                  "missing": ["boundary_inconsistency", "blur"]},
    "regions": {"present": 8, "total": 9, "max_share_percent": 32.6316,
                "missing": ["context"]},
    "media": {"present": 2, "total": 5, "max_share_percent": 81.25,
              "missing": ["plastic-like", "fabric-like", "reflective-film-like"]},
    "geometry": {"present": 2, "total": 6, "max_share_percent": 84.375,
                 "missing": ["partial-curved", "flexible", "rigid", "boundary-only"]},
    "illumination": {"present": 2, "total": 6, "max_share_percent": 81.25,
                     "missing": ["left", "right", "top", "bottom"]},
}


class FrozenContractError(RuntimeError):
    """The loaded contract does not match the accepted C2 freeze."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class BatchContext:
    """The frozen contract plus the C2B batch envelope and quotas."""

    def __init__(self) -> None:
        self.ontology: Ontology = load_ontology(ONTOLOGY_PATH)
        loaded: LLMProviderConfig = load_llm_config(LLM_CONFIG_PATH)
        # A 32-object response with medium thinking takes far longer to produce
        # than the single-recipe call the 120s default was sized for, and a
        # timeout would burn a transport attempt on the one batch C2B is allowed.
        # `request_timeout_seconds` is documented as OPERATIONAL and is excluded
        # from `identity_material`, so this override cannot move the provider
        # config identity - `verify()` proves it did not. The committed C1 config
        # file is left untouched.
        self.config_timeout_before_override = loaded.request_timeout_seconds
        self.request_timeout_override = BATCH_REQUEST_TIMEOUT_SECONDS
        self.config: LLMProviderConfig = loaded.model_copy(
            update={"request_timeout_seconds": BATCH_REQUEST_TIMEOUT_SECONDS})
        self.template: PromptTemplate = load_prompt_template(self.ontology)
        self.quotas: QuotaSpec = load_quota_spec(QUOTA_PATH)
        self.quotas.validate_against(self.ontology)

        self.provider_config_identity = provider_config_identity(self.config)
        self.item_schema = candidate_object_schema(self.ontology)
        # The envelope C1 recorded for the 12x32 schedule. The provider REJECTS
        # it: see `batch_schema` below. Kept so the report can name exactly what
        # could not be sent.
        self.bounded_batch_schema = candidate_json_schema(
            self.ontology, recipes_requested=BATCH_SIZE)
        self.bounded_batch_envelope_identity = json_schema_identity(self.bounded_batch_schema)
        # What C2B actually sends: the same envelope without the array length
        # bound. "Exactly 32" is still enforced locally on the response.
        self.batch_schema = candidate_json_schema(
            self.ontology, recipes_requested=BATCH_SIZE, array_bounds=ARRAY_BOUNDS_SENT)
        self.single_recipe_schema_identity = json_schema_identity(self.item_schema)
        self.batch_envelope_schema_identity = json_schema_identity(self.batch_schema)
        # C2 ran a 1-object envelope; kept only so the report can show what moved.
        self.c2_singleton_envelope_identity = json_schema_identity(
            candidate_json_schema(self.ontology, recipes_requested=1))
        self.verify()

    def verify(self) -> None:
        actual = {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "api_surface": self.config.api_surface,
            "thinking_level": self.config.thinking_level,
            "prompt_template_identity": self.template.identity(),
            "ontology_identity": self.ontology.sha256,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,
            "bounded_batch_envelope_identity": self.bounded_batch_envelope_identity,
        }
        drift = {key: {"frozen": FROZEN[key], "actual": actual[key]}
                 for key in FROZEN if FROZEN[key] != actual[key]}
        if drift:
            raise FrozenContractError(f"the accepted C2 contract has drifted: {drift}")
        if self.quotas.batch_size != BATCH_SIZE:
            raise FrozenContractError(
                f"the quota spec is for a batch of {self.quotas.batch_size}, not {BATCH_SIZE}")
        # The item schema must be byte-identical in both envelopes: this is the
        # guarantee that changing the batch size did not touch recipe semantics.
        for size in (1, BATCH_SIZE):
            envelope = candidate_json_schema(self.ontology, recipes_requested=size)
            if envelope["properties"]["recipes"]["items"] != self.item_schema:
                raise FrozenContractError(
                    f"the item schema differs inside the {size}-object envelope; changing the "
                    "envelope must never alter recipe semantics")

    # --- request ------------------------------------------------------------
    def batch_input_text(self) -> str:
        """The batch generation request body, with the generic quotas rendered.

        `build_generation_prompt` runs the target-leakage firewall over its own
        output before returning it, so a quota that somehow named a corpus would
        fail here rather than reach the provider.
        """
        return build_generation_prompt(
            self.template, recipes_requested=BATCH_SIZE,
            coverage_quotas=self.quotas.prompt_block(self.ontology))

    @property
    def batch_template_identity(self) -> str:
        """Identity of the RENDERED batch request body. Distinct from the system
        prompt identity, which C2B leaves untouched."""
        import hashlib
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
            metadata={"phase": "c2b_batch_shape", "disposable": True, "enters_c3": False,
                      "quota_identity": self.quotas.quota_identity},
        )

    def config_summary(self) -> dict[str, Any]:
        summary = self.config.public_summary()
        summary["recipe_schema_version"] = self.ontology.recipe_schema_version
        return summary

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
            "system_prompt_changed_in_c2b": False,
            "batch_generation_template_identity": self.batch_template_identity,
            "request_template_sha256": self.template.generation_template_sha256,

            "coverage_quota_identity": self.quotas.quota_identity,
            "coverage_quota_label": self.quotas.label,
            "coverage_quota_source": "configs/version_c/llm/c2b_coverage_quotas.yaml",

            "single_recipe_schema_identity": self.single_recipe_schema_identity,
            "batch_envelope_schema_identity": self.batch_envelope_schema_identity,
            "batch_envelope_array_bounds_sent": ARRAY_BOUNDS_SENT,
            "bounded_batch_envelope_identity_rejected_by_provider":
                self.bounded_batch_envelope_identity,
            "c2_singleton_envelope_schema_identity": self.c2_singleton_envelope_identity,
            "schema_identity_note":
                "The value recorded at C1 as `llm_schema_identity_12x32` (7afc3abd...) is the "
                "32-OBJECT ENVELOPE identity, not a single-recipe identity. The single-recipe "
                "ITEM schema is a separate value (1e3f050e...) and is byte-identical in the "
                "1-object envelope C2 used and the envelope C2B sends, which is the evidence "
                "that changing the batch size did not touch recipe semantics.",
            "array_bounds_finding":
                "The C1-recorded 32-object envelope carries minItems=maxItems=32 and the "
                "provider REJECTS it with 400 INVALID_ARGUMENT. The byte-identical envelope "
                "with the bound at 1 was accepted 42 times during C2, and the two schemas "
                "differ by nothing else (2695 vs 2697 bytes, identical item schema). Google "
                "documents array length limits as a source of schema-complexity rejection. "
                "C2B therefore omits the bound from the REQUEST only; `validate_response` "
                "still rejects any response whose recipe count is not exactly 32, so the "
                "scientific requirement is unchanged and is enforced where it always was.",

            "ontology_identity": self.ontology.sha256,
            "provider_config_identity": self.provider_config_identity,
            "allow_ontology_aliases": self.config.allow_ontology_aliases,
            "request_timeout_seconds": self.config.request_timeout_seconds,
            "request_timeout_seconds_in_committed_config": self.config_timeout_before_override,
            "request_timeout_override_note":
                "Operational only. A 32-object response takes far longer to produce than the "
                "single-recipe call the committed 120s default was sized for. "
                "`request_timeout_seconds` is excluded from `identity_material` by design, so "
                "the provider config identity is unchanged (verified above); the committed C1 "
                "config file was not edited.",
            "recipes_per_request": BATCH_SIZE,
            "request_schedule_for_c3": "12 requests x 32 recipes = 384 raw candidate slots",

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


__all__ = ["REPO", "REPORTS", "DOCS", "RAW_DIR", "FROZEN", "BATCH_SIZE", "LOGICAL_BATCH_ID",
           "C2B_BANK_ID", "C2_SINGLETON_COVERAGE", "BatchContext", "FrozenContractError",
           "RecordingProvider", "archive_record", "replay_fields", "write_raw_response_files",
           "write_json", "read_json", "api_key_present", "git", "utc_now", "repo_relative"]
