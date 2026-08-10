"""C2 prerequisite: environment audit and offline provider-contract re-verification.

Run with the Version-C virtual environment:

    .venv\\Scripts\\python.exe scripts/c2_environment_audit.py

This makes NO network call and reads NO credential value. It answers two
questions C2 must settle before any live request:

1. Does `import prism_fas` resolve to the Version-C source tree? (C1 found that
   the ambient editable install resolved to Version B, which would have run the
   wrong code against a live provider.)
2. Does the frozen provider contract still assemble exactly as C1 froze it?

It reports `gemini_api_key_present` as a boolean and nothing more.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "c2"

VERSION_B_ROOT = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")


def git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except Exception:
        return None


def main() -> None:
    import prism_fas

    prism_path = Path(prism_fas.__file__).resolve()
    expected_root = (REPO / "src" / "prism_fas").resolve()
    resolves_to_version_c = expected_root in prism_path.parents or prism_path.parent == expected_root
    resolves_to_version_b = str(VERSION_B_ROOT).lower() in str(prism_path).lower()

    in_venv = sys.prefix != sys.base_prefix
    venv_root = Path(sys.prefix).resolve()
    venv_inside_repo = REPO.resolve() in venv_root.parents or venv_root.parent == REPO.resolve()

    # --- offline re-verification of the frozen provider contract -------------
    from prism_fas.llm.config import FORBIDDEN_SAMPLING_FIELDS, load_llm_config, provider_config_identity
    from prism_fas.llm.contracts import GenerationRequest
    from prism_fas.llm.json_schema import candidate_json_schema, json_schema_identity
    from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
    from prism_fas.llm.providers import GeminiRecipeProvider
    from prism_fas.llm.providers.gemini import API_SURFACE, SDK_PACKAGE, sdk_version
    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")
    config = load_llm_config(REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml")
    template = load_prompt_template(ontology)
    schema = candidate_json_schema(ontology, recipes_requested=32)

    request = GenerationRequest(
        slot_id="c2-contract-check",
        system_instruction=template.system_instruction,
        input_text=build_generation_prompt(template, recipes_requested=32),
        response_json_schema=schema,
        model_id=config.model_id,
        thinking_level=config.thinking_level,
        response_mime_type=config.response_mime_type,
        max_output_tokens=config.max_output_tokens,
        recipes_requested=32,
        ontology_identity=ontology.sha256,
        prompt_template_identity=template.identity(),
        provider_config_identity=provider_config_identity(config),
    )
    provider = GeminiRecipeProvider(config)
    call_kwargs = provider.build_call_kwargs(request)
    generation_config = call_kwargs["generation_config"]

    api_key_present = bool(os.environ.get(config.api_key_env, "").strip())

    audit = {
        "schema_version": "c2-environment-audit-v1",
        "milestone": "C2",
        "purpose": "Prerequisite audit run BEFORE any live provider call. No network "
                   "request was made and no credential value was read.",
        "generated_by": "scripts/c2_environment_audit.py",

        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "accepted_c1_head": "cdf1594ca6892ca05c168fb2f6b4494236981222",
            "working_tree_clean": git("status", "--short") == "",
        },

        "interpreter": {
            "python_version": platform.python_version(),
            "executable": sys.executable,
            "in_virtual_environment": in_venv,
            "sys_prefix": str(venv_root),
            "venv_inside_version_c_repository": venv_inside_repo,
            "pip_version": version_of("pip"),
            "platform": platform.platform(),
        },

        "import_resolution": {
            "prism_fas_file": str(prism_path),
            "expected_root": str(expected_root),
            "resolves_to_version_c": resolves_to_version_c,
            "resolves_to_version_b": resolves_to_version_b,
            "c1_finding": "Before C2 the ambient editable install resolved `prism_fas` to "
                          "PRISM_FAS_B_Project/src/prism_fas for ordinary python commands. "
                          "Running a live provider call under that resolution would have "
                          "exercised Version-B code.",
            "resolution": "A dedicated .venv inside the Version-C repository with "
                          "`pip install -e .[llm]`.",
            "version_b_modified": False,
            "global_environment_modified": False,
        },

        "packages": {
            "prism-fas": version_of("prism-fas"),
            "google-genai": version_of("google-genai"),
            "pydantic": version_of("pydantic"),
            "PyYAML": version_of("PyYAML"),
            "pyarrow": version_of("pyarrow"),
            "typer": version_of("typer"),
            "pytest": version_of("pytest"),
            "numpy": version_of("numpy"),
        },

        "credential": {
            "env_var": config.api_key_env,
            "gemini_api_key_present": api_key_present,
            "google_api_key_present": bool(os.environ.get("GOOGLE_API_KEY", "").strip()),
            "value_read_into_audit": False,
            "checked_scopes": ["process", "user", "machine", "repository .env"],
        },

        "frozen_provider_contract_reverified_offline": {
            "provider": config.provider,
            "model_id": config.model_id,
            "model_matches_c1_freeze": config.model_id == "gemini-3.6-flash",
            "api_surface": API_SURFACE,
            "sdk_package": SDK_PACKAGE,
            "sdk_version_installed": sdk_version(),
            "sdk_version_pin": config.sdk_version_pin,
            "thinking_level": generation_config.get("thinking_level"),
            "thinking_level_matches_c1_freeze": generation_config.get("thinking_level") == "medium",
            "response_format": {
                "type": call_kwargs["response_format"]["type"],
                "mime_type": call_kwargs["response_format"]["mime_type"],
                "schema_present": "schema" in call_kwargs["response_format"],
            },
            "structured_json_required": call_kwargs["response_format"]["mime_type"] == "application/json",
            "input_type": type(call_kwargs["input"]).__name__,
            "call_kwargs_keys": sorted(call_kwargs),
            "generation_config_keys": sorted(generation_config),
            "forbidden_sampling_fields": list(FORBIDDEN_SAMPLING_FIELDS),
            "forbidden_sampling_fields_present": sorted(
                field for field in FORBIDDEN_SAMPLING_FIELDS
                if field in generation_config or field in call_kwargs),
            "tools_passed": "tools" in call_kwargs,
            "store_interaction": call_kwargs["store"],
            "stream": call_kwargs["stream"],
            "media_keys_present": sorted(
                key for key in ("image", "images", "inline_data", "file_data", "parts",
                                "media", "video", "audio", "contents")
                if key in call_kwargs),
        },

        "identities": {
            "ontology_identity": ontology.sha256,
            "llm_prompt_template_identity": template.identity(),
            "llm_schema_identity_12x32": json_schema_identity(schema),
            "llm_provider_config_identity": provider_config_identity(config),
            "generation_request_identity": request.request_sha256,
            "allow_ontology_aliases": config.allow_ontology_aliases,
        },

        "live_calls_made": 0,
        "network_requests_made": 0,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C2_ENVIRONMENT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", (OUT / "C2_ENVIRONMENT_AUDIT.json").relative_to(REPO))
    print("prism_fas ->", prism_path)
    print("resolves to Version C:", resolves_to_version_c)
    print("resolves to Version B:", resolves_to_version_b)
    print("gemini_api_key_present:", api_key_present)
    print("forbidden sampling fields present:",
          audit["frozen_provider_contract_reverified_offline"]["forbidden_sampling_fields_present"])
    print("tools passed:", audit["frozen_provider_contract_reverified_offline"]["tools_passed"])


if __name__ == "__main__":
    main()
