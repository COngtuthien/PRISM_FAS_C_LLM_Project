"""PRISM-FAS-C LLM recipe-planner contract (milestone C1).

The LLM is an OFFLINE SEMANTIC RECIPE PLANNER. It is not the FAS classifier, not
an image generator, not a target-image interpreter, not part of inference and not
part of the training loop. It runs once, before training, and its output is frozen.

Nothing in this package generates a scientific recipe bank. C1 builds the
contract; C2 runs a 32-recipe disposable pilot; C3 runs the frozen 384-slot
scientific generation. Those are separate milestones with their own acceptance.

Everything downstream depends on `RecipeProvider`, never on a vendor SDK.
"""
from __future__ import annotations

from .config import (
    LLMProviderConfig,
    LLMConfigError,
    load_llm_config,
    provider_config_identity,
)
from .contracts import (
    ErrorClass,
    GenerationRequest,
    ProviderError,
    ProviderGenerationResult,
    RETRYABLE_ERROR_CLASSES,
    NON_RETRYABLE_ERROR_CLASSES,
)
from .firewall import (
    FirewallViolation,
    RequestFirewallError,
    assert_request_is_target_free,
    redact_secrets,
    scan_for_target_leakage,
)
from .json_schema import candidate_json_schema, json_schema_identity
from .prompt import (
    PromptTemplate,
    build_generation_prompt,
    load_prompt_template,
    prompt_template_identity,
)
from .provenance import (
    REQUIRED_PROVENANCE_FIELDS,
    GenerationProvenance,
    ProvenanceError,
    identity_chain,
)
from .pipeline import (
    CandidateOutcome,
    RecipeCandidateResult,
    RecipePlanner,
    QuotaBlocked,
    QuotaState,
)
from .providers import (
    GeminiRecipeProvider,
    MockRecipeProvider,
    RecipeProvider,
    ReplayRecipeProvider,
)

__all__ = [
    "LLMProviderConfig", "LLMConfigError", "load_llm_config", "provider_config_identity",
    "ErrorClass", "GenerationRequest", "ProviderError", "ProviderGenerationResult",
    "RETRYABLE_ERROR_CLASSES", "NON_RETRYABLE_ERROR_CLASSES",
    "FirewallViolation", "RequestFirewallError", "assert_request_is_target_free",
    "redact_secrets", "scan_for_target_leakage",
    "candidate_json_schema", "json_schema_identity",
    "PromptTemplate", "build_generation_prompt", "load_prompt_template", "prompt_template_identity",
    "REQUIRED_PROVENANCE_FIELDS", "GenerationProvenance", "ProvenanceError", "identity_chain",
    "CandidateOutcome", "RecipeCandidateResult", "RecipePlanner", "QuotaBlocked", "QuotaState",
    "GeminiRecipeProvider", "MockRecipeProvider", "RecipeProvider", "ReplayRecipeProvider",
]
