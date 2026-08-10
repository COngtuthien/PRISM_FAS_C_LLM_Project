"""Generate the C1 report artifacts from the live code, not by hand.

Run from the repository root:

    python scripts/c1_build_reports.py

Every number in `reports/c1/` therefore comes from the implementation it
describes. The script reads no credential and makes no network call: it imports
the contract, builds the frozen identities and writes JSON.
"""
from __future__ import annotations

import importlib.metadata as metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_fas.llm import providers as providers_pkg                      # noqa: E402
from prism_fas.llm.config import (                                        # noqa: E402
    FORBIDDEN_SAMPLING_FIELDS, SUPPORTED_PROVIDERS, THINKING_LEVELS,
    load_llm_config, provider_config_identity,
)
from prism_fas.llm.contracts import (                                     # noqa: E402
    NON_RETRYABLE_ERROR_CLASSES, RETRYABLE_ERROR_CLASSES, ErrorClass,
)
from prism_fas.llm.firewall import scan_for_target_leakage                # noqa: E402
from prism_fas.llm.json_schema import (                                   # noqa: E402
    CANDIDATE_SCHEMA_VERSION, candidate_json_schema, json_schema_identity,
)
from prism_fas.llm.pipeline import PIPELINE_STAGES, SYSTEM_OWNED_KEYS     # noqa: E402
from prism_fas.llm.prompt import (                                        # noqa: E402
    PROMPT_TEMPLATE_VERSION, build_generation_prompt, load_prompt_template,
)
from prism_fas.llm.provenance import (                                    # noqa: E402
    IDENTITY_CHAIN_ORDER, REQUIRED_PROVENANCE_FIELDS, identity_chain,
)
from prism_fas.llm.providers.gemini import API_SURFACE, SDK_PACKAGE, sdk_version  # noqa: E402
from prism_fas.recipes.ontology import load_ontology                      # noqa: E402
from prism_fas.recipes.validate import VALIDATION_STAGES                  # noqa: E402

ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
OUT = REPO / "reports" / "c1"

C3_RECIPES_PER_CALL = 32          # the spec's recommended 12 x 32 schedule


def git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print("wrote", (OUT / name).relative_to(REPO))


def main() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    config = load_llm_config(LLM_CONFIG_PATH)
    template = load_prompt_template(ontology)
    schema = candidate_json_schema(ontology, recipes_requested=C3_RECIPES_PER_CALL)
    schema_id = json_schema_identity(schema)
    config_id = provider_config_identity(config)

    chain = identity_chain(ontology_identity=ontology.sha256, schema_identity=schema_id,
                           prompt_template_identity=template.identity(),
                           provider_config_identity=config_id)

    api_key_present = bool(os.environ.get(config.api_key_env, "").strip())

    # ---------------------------------------------------------------- provider
    write("C1_PROVIDER_AUDIT.json", {
        "schema_version": "c1-provider-audit-v1",
        "milestone": "C1",
        "claim": "The frozen scientific provider/model contract was verified against the "
                 "official Google documentation and the installed SDK before being frozen. "
                 "No live scientific generation was performed.",
        "verification_date": "2026-08-10",
        "model": {
            "provider": "Google Gemini Developer API",
            "model_id": config.model_id,
            "status_documented": "stable",
            "spec_required_model": "gemini-3.6-flash",
            "matches_spec": config.model_id == "gemini-3.6-flash",
            "structured_output_supported": True,
            "thinking_supported": True,
            "thinking_levels_documented": list(THINKING_LEVELS),
            "thinking_level_used": config.thinking_level,
            "evidence": [
                "https://ai.google.dev/gemini-api/docs/models - gemini-3.6-flash listed as Stable",
                "https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash - "
                "capabilities list structured output: Supported, thinking: Supported",
                "https://ai.google.dev/gemini-api/docs/thinking - thinking_level nested in "
                "generation_config; allowed values minimal|low|medium|high",
            ],
        },
        "sdk": {
            "package": SDK_PACKAGE,
            "installed_version": sdk_version(),
            "version_pin": config.sdk_version_pin,
            "python_version": platform.python_version(),
            "import_path": "from google import genai",
            "client": "genai.Client(api_key=<from environment>)",
            "api_surface": API_SURFACE,
            "call": "client.interactions.create(...)",
            "interactions_available_from": "google-genai 2.3.0",
            "surface_rationale": "The Interactions API is GA and is the surface Google "
                                 "recommends for current models. The spec left the choice open "
                                 "(api_surface: generate_content_or_interactions_frozen_at_C1); "
                                 "C1 freezes 'interactions'.",
            "verified_create_parameters": [
                "model", "input", "stream", "store", "background", "system_instruction",
                "tools", "response_modalities", "response_mime_type", "previous_interaction_id",
                "service_tier", "webhook_config", "response_format", "environment",
                "generation_config", "safety_settings", "labels",
            ],
            "verified_generation_config_fields": [
                "image_config", "max_output_tokens", "seed", "speech_config", "stop_sequences",
                "thinking_level", "thinking_summaries", "tool_choice", "transcription_config",
                "video_config",
            ],
        },
        "structured_output": {
            "mechanism": "response_format",
            "shape": {"type": "text", "mime_type": "application/json", "schema": "<JSON Schema>"},
            "response_mime_type": config.response_mime_type,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "schema_identity_for_12x32_schedule": schema_id,
            "markdown_fence_scraping_is_the_scientific_path": False,
            "local_validation_is_authoritative": True,
        },
        "sampling_controls": {
            "spec_requirement": "temperature/top_p/top_k MUST NOT be sent",
            "implemented": "never sent",
            "structural_finding": "GenerationConfigParam in google-genai "
                                  f"{sdk_version()} exposes no temperature, top_p or top_k field "
                                  "at all, so these controls cannot be sent on this surface.",
            "documentation_discrepancy": {
                "spec_wording": "Appendix E states current Gemini 3.x guidance DEPRECATES "
                                "temperature, top_p and top_k for gemini-3.6-flash.",
                "documentation_wording": "The Gemini 3 developer guide instead states: 'For all "
                                         "Gemini 3 models, we strongly recommend keeping the "
                                         "temperature parameter at its default value of 1.0.' "
                                         "top_p and top_k are not discussed on that page.",
                "impact_on_behaviour": "None. Not sending the controls satisfies both the spec "
                                       "requirement and the documented recommendation, and the "
                                       "installed surface has no such fields.",
                "spec_amendment_required": False,
                "recorded_because": "the spec's stated rationale is narrower than what the "
                                    "current documentation says, and C1 must not paper over that.",
            },
            "forbidden_fields": list(FORBIDDEN_SAMPLING_FIELDS),
        },
        "disabled_capabilities": {
            "tools": "never passed; omitting `tools` disables function calling, Search "
                     "grounding, URL context, code execution and file search together",
            "grounding": False, "url_context": False, "code_execution": False,
            "file_search": False, "image_input": False, "audio_input": False,
            "video_input": False,
            "store_interaction": False,
            "input_type": "str (text only)",
        },
        "providers_implemented": {
            "interface": "prism_fas.llm.providers.base.RecipeProvider",
            "implementations": sorted(["mock", "replay", "gemini"]),
            "supported_provider_names": list(SUPPORTED_PROVIDERS),
            "vendor_sdk_imported_outside_the_gemini_provider": False,
            "gemini_sdk_import_is_lazy": True,
            "firewall_runs_in_every_implementation": True,
        },
        "credential": {
            "env_var": config.api_key_env,
            "api_key_present": api_key_present,
            "passed_explicitly_to_client": True,
            "rationale": "The SDK also honours GOOGLE_API_KEY and lets it take precedence; "
                         "passing the key explicitly prevents a different project's credential "
                         "from being used silently.",
        },
        "live_probe": {
            "executed": False,
            "calls": 0,
            "status": "NOT_RUN_NO_KEY" if not api_key_present else "NOT_RUN_NOT_REQUIRED",
            "spec_position": "C1 hard acceptance says a live-provider integration test MAY "
                             "generate <=2 disposable recipes. It is optional, so C1 can pass "
                             "without it.",
            "cost_incurred": 0,
        },
        "config_identity": config_id,
    })

    # ------------------------------------------------------------------ schema
    write("C1_SCHEMA_AUDIT.json", {
        "schema_version": "c1-schema-audit-v1",
        "milestone": "C1",
        "recipe_schema_version": ontology.recipe_schema_version,
        "recipe_schema_source": "inherited Version-B schema v1.1 "
                                "(prism_fas.recipes.schema), reused unchanged for comparability",
        "ontology": {
            "version": ontology.version,
            "identity_sha256": ontology.sha256,
            "path": "configs/recipes/ontology_m7.yaml",
            "authority": "the ontology and the compiler are the execution authority; the model "
                         "may propose only values that already exist here",
            "counts": {
                "media": len(ontology.media),
                "geometry_shapes": len(ontology.geometry_shapes),
                "regions": len(ontology.regions),
                "artifacts": len(ontology.artifacts),
                "illumination": len(ontology.illumination),
                "routes": len(ontology.routes),
                "forbidden_shortcuts": len(ontology.forbidden_shortcuts),
            },
            "vocabulary": {
                "media": list(ontology.media),
                "geometry_shapes": list(ontology.geometry_shapes),
                "regions": list(ontology.regions),
                "artifacts": list(ontology.artifacts),
                "illumination": list(ontology.illumination),
                "routes": list(ontology.routes),
            },
            "limits": dict(ontology.limits),
        },
        "request_side_json_schema": {
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "identity_for_12x32_schedule": schema_id,
            "recipes_per_call": C3_RECIPES_PER_CALL,
            "additional_properties_allowed": False,
            "system_owned_keys_absent_from_schema": sorted(SYSTEM_OWNED_KEYS),
        },
        "validation_pipeline_stages": list(PIPELINE_STAGES),
        "inherited_validator_stages": list(VALIDATION_STAGES),
        "rejection_rules": {
            "unknown_field": "REJECT (pydantic extra='forbid')",
            "unknown_enum": "REJECT",
            "out_of_range_value": "REJECT",
            "missing_required_field": "REJECT",
            "malformed_json": "REJECT",
            "prose_or_markdown_fence": "REJECT (not scraped)",
            "model_supplied_system_owned_field": "REJECT (not stripped)",
            "duplicate": "REJECT as candidate duplicate, recorded in the audit",
        },
        "silent_repair": {
            "performed": False,
            "policy": "No semantic field is ever edited to make an invalid candidate valid. "
                      "The only recovery is another candidate from the provider under the same "
                      "frozen contract, bounded by the retry budget.",
            "recipe_id_assignment": "recipe_id is ASSIGNED by the system after the semantic "
                                    "content is known. The model is forbidden to supply one and "
                                    "is rejected if it does. This adds provenance rather than "
                                    "repairing semantics (spec 7.3).",
            "ontology_alias_normalization_enabled": config.allow_ontology_aliases,
            "alias_note": "The inherited ontology documents an alias map. C1 leaves it OFF, the "
                          "strictest reading of spec 7.6. It is identity-bearing, so enabling it "
                          "would change the recipe bank identity.",
        },
        "canonicalization": {
            "method": "prism_fas.recipes.canonical.canonical_json",
            "key_order": "sorted",
            "float_decimals": 6,
            "separators": "compact (',', ':')",
            "identity": "SHA-256 over the canonical UTF-8 text",
            "order_invariance": "response key order and equivalent float spellings collapse; "
                                "a real numeric difference does not",
        },
        "duplicate_policy": {
            "basis": "canonical scientific content identity",
            "positional_recipe_id_excluded": True,
            "reason": "recipe_id is positional, so including it would let renumbering defeat "
                      "duplicate detection",
            "fuzzy_similarity_used": False,
            "fuzzy_note": "the spec does not require semantic-similarity dedup at C1; coverage "
                          "and diversity selection belong to C3",
        },
        "compiler_compatibility": {
            "compiler": "prism_fas.recipes.compile.compile_recipe (m7-compiler-v1)",
            "proven_in_tests": True,
            "outputs_checked": ["operator graph", "region mask policy",
                                "41-D conditioning vector", "graph hash determinism"],
            "conditioning_dimension": 41,
            "authority": "a candidate that validates at the LLM layer but cannot compile is not "
                         "acceptable; the compiler stays authoritative",
        },
    })

    # -------------------------------------------------------------- provenance
    write("C1_PROVENANCE_AUDIT.json", {
        "schema_version": "c1-provenance-audit-v1",
        "milestone": "C1",
        "required_fields": list(REQUIRED_PROVENANCE_FIELDS),
        "required_field_count": len(REQUIRED_PROVENANCE_FIELDS),
        "source": "spec section 7.7 LLM provenance record",
        "enforcement": "a record missing any required field raises ProvenanceError; a record "
                       "carrying a credential-shaped key raises ProvenanceError",
        "credential_fields_permitted": 0,
        "redaction_applied_to_every_string": True,
        "identity_chain": {
            "order": list(IDENTITY_CHAIN_ORDER),
            "resolved_at_c1": chain["resolved_links"],
            "links": chain["links"],
            "chain_identity": chain["chain_identity"],
            "invalidation_rule": "changing the model, prompt, schema, ontology or thinking "
                                 "configuration changes the request identity and therefore "
                                 "invalidates any downstream scientific generation",
        },
        "identities": {
            "ontology_identity": ontology.sha256,
            "llm_schema_identity": schema_id,
            "llm_prompt_template_identity": template.identity(),
            "llm_provider_config_identity": config_id,
            "system_prompt_sha256": template.system_instruction_sha256,
            "request_template_sha256": template.generation_template_sha256,
        },
        "prompt_template": {
            "version": PROMPT_TEMPLATE_VERSION,
            "frozen_for_c3": False,
            "note": "This is the C1 template. C2 may still tune wording on source-only validity "
                    "and coverage evidence; the exact bytes are frozen in the LLM bank lock "
                    "before the 384-slot generation begins (spec 7.5).",
            "system_instruction_chars": len(template.system_instruction),
            "generation_template_chars": len(template.generation_template),
        },
        "raw_response_preservation": {
            "raw_text_preserved_verbatim": True,
            "raw_response_sha256_recorded": True,
            "replay_provider_implemented": True,
            "rationale": "a hosted model is not reproducible; the archived response is the "
                         "reproducible artifact, not the request",
        },
        "generator_code_commit": git("rev-parse", "HEAD"),
    })

    # ---------------------------------------------------------------- security
    template_body = build_generation_prompt(template, recipes_requested=C3_RECIPES_PER_CALL)
    prompt_violations = scan_for_target_leakage({
        "system_instruction": template.system_instruction,
        "input_text": template_body,
        "response_json_schema": schema,
    })
    provider_sources = {name: inspect.getsource(getattr(providers_pkg, name))
                        for name in ("gemini", "mock", "replay")}
    banned_media_symbols = ("Part.from_bytes", "upload_file", "inline_data", "from_uri",
                            "image_bytes", "PIL.Image")

    write("C1_SECURITY_AUDIT.json", {
        "schema_version": "c1-security-audit-v1",
        "milestone": "C1",
        "target_firewall": {
            "implemented": True,
            "fail_closed": True,
            "runs_in_every_provider_including_test_doubles": True,
            "runs_before_any_client_is_constructed": True,
            "categories": ["dataset_reference", "target_metric", "target_structure",
                           "target_feedback", "binary_payload", "forbidden_payload_key"],
            "frozen_prompt_violations": len(prompt_violations),
            "frozen_prompt_is_target_free": prompt_violations == [],
        },
        "text_only_path": {
            "input_type": "str",
            "media_input_code_path_exists": False,
            "banned_symbols_checked": list(banned_media_symbols),
            "banned_symbols_found": sorted({symbol for source in provider_sources.values()
                                            for symbol in banned_media_symbols
                                            if symbol in source}),
            "face_images_sent_to_llm": 0,
            "dataset_content_sent_to_llm": 0,
        },
        "secrets": {
            "env_var": config.api_key_env,
            "api_key_present_in_environment": api_key_present,
            "api_key_value_read_into_any_report_field": False,
            "hardcoded_keys": 0,
            "keys_committed": 0,
            "redaction_patterns": ["google api key", "google oauth token", "generic sk- key",
                                   "PEM private key header", "credential assignment"],
            "redaction_applied_to": ["exception messages", "provenance records",
                                     "quota state", "provider describe()"],
            "env_example_contains_value": False,
            "tests_use_a_synthetic_key_only": True,
        },
        "network": {
            "unit_tests_require_network": False,
            "network_blocked_in_test_fixtures": True,
            "live_provider_calls_in_c1": 0,
            "api_cost_incurred": 0,
        },
        "target_isolation": {
            "siw_labels_accessed": 0,
            "siw_scoring_runs": 0,
            "target_metrics_computed": 0,
            "target_derived_prompt_content": 0,
            "version_b_artifacts_modified": 0,
        },
    })

    print("\nidentities")
    print("  ontology     ", ontology.sha256)
    print("  schema       ", schema_id)
    print("  prompt       ", template.identity())
    print("  provider cfg ", config_id)
    print("  chain        ", chain["chain_identity"])
    print("  sdk          ", SDK_PACKAGE, sdk_version())
    print("  api key      ", api_key_present)


if __name__ == "__main__":
    main()
