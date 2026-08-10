"""The C1 validation pipeline, retry orchestration and quota state machine.

    raw provider response
        -> JSON parsing
        -> JSON Schema envelope validation
        -> typed recipe validation      (prism_fas.recipes.schema, extra="forbid")
        -> ontology membership          (prism_fas.recipes.validate)
        -> range checks
        -> compatibility checks
        -> canonicalization             (prism_fas.recipes.canonical)
        -> recipe identity              (SHA-256 of the canonical text)
        -> ACCEPT / REJECT

The LLM cannot override the validator. Nothing in this module edits the semantic
content of a candidate to make it valid: an invalid candidate is rejected and
recorded, and the only permitted recovery is asking the provider for another
candidate under the same frozen contract.

`recipe_id` is assigned by the system after the semantic content is known, which
is the spec's rule that the id, the content hash and the validation status are
computed rather than trusted from the model. That assignment is not a repair: it
adds provenance the model was forbidden to supply.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RECIPE_ID_PATTERN, RecipeV11
from prism_fas.recipes.validate import ValidationIssue, canonicalize_payload, validate_payload

from .config import LLMProviderConfig
from .contracts import ErrorClass, GenerationRequest, ProviderError, ProviderGenerationResult
from .providers.base import RecipeProvider

#: Ordered pipeline stages, recorded in the schema audit.
PIPELINE_STAGES: tuple[str, ...] = (
    "json_parsing",
    "envelope_schema",
    "typed_recipe_schema",
    "ontology_membership",
    "range_checks",
    "compatibility_checks",
    "canonicalization",
    "recipe_identity",
    "duplicate_detection",
)

#: Keys the model is forbidden to supply because the system computes them.
SYSTEM_OWNED_KEYS: frozenset[str] = frozenset({
    "recipe_id", "recipe_hash", "content_hash", "identity", "provenance",
    "validation", "validation_result", "valid", "accepted",
})


class CandidateOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_JSON = "rejected_json"
    REJECTED_ENVELOPE = "rejected_envelope"
    REJECTED_SCHEMA = "rejected_schema"
    REJECTED_ONTOLOGY = "rejected_ontology"
    REJECTED_DUPLICATE = "rejected_duplicate"
    REJECTED_SYSTEM_OWNED_FIELD = "rejected_system_owned_field"


@dataclass(frozen=True)
class RecipeCandidateResult:
    """One candidate object from one response."""

    index: int
    outcome: CandidateOutcome
    recipe: RecipeV11 | None = None
    canonical_text: str | None = None
    recipe_identity: str | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.outcome is CandidateOutcome.ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "outcome": self.outcome.value,
                "recipe_identity": self.recipe_identity,
                "recipe_id": self.recipe.recipe_id if self.recipe is not None else None,
                "issue_count": len(self.issues), "issues": self.issues}


@dataclass(frozen=True)
class ResponseValidation:
    """The verdict on one whole provider response."""

    candidates: list[RecipeCandidateResult]
    response_issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> list[RecipeCandidateResult]:
        return [item for item in self.candidates if item.accepted]

    @property
    def all_accepted(self) -> bool:
        return bool(self.candidates) and not self.response_issues and all(
            item.accepted for item in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_count": len(self.candidates),
                "accepted_count": len(self.accepted),
                "all_accepted": self.all_accepted,
                "response_issues": self.response_issues,
                "candidates": [item.as_dict() for item in self.candidates]}


class QuotaBlocked(RuntimeError):
    """Daily quota is exhausted. Completed slots are preserved; the run stops.

    Never raised for a transient rate limit: that path backs off and retries the
    same frozen request.
    """

    def __init__(self, state: "QuotaState") -> None:
        self.state = state
        super().__init__("provider daily quota exhausted; completed slots preserved, run stopped")


@dataclass
class QuotaState:
    """What a resume needs to know. Written to `API_QUOTA_BLOCKED.json`."""

    blocked: bool = False
    reason: str | None = None
    blocked_at_utc: str | None = None
    completed_slot_ids: list[str] = field(default_factory=list)
    pending_slot_ids: list[str] = field(default_factory=list)
    billing_tier: str = "free"

    def as_dict(self) -> dict[str, Any]:
        return {"blocked": self.blocked, "reason": self.reason,
                "blocked_at_utc": self.blocked_at_utc,
                "completed_slot_ids": sorted(self.completed_slot_ids),
                "pending_slot_ids": sorted(self.pending_slot_ids),
                "completed_count": len(self.completed_slot_ids),
                "pending_count": len(self.pending_slot_ids),
                "billing_tier": self.billing_tier,
                "auto_enable_paid": False,
                "resume_policy": "resume only the pending slot ids under the same frozen "
                                 "model, prompt, schema and selection algorithm; never "
                                 "regenerate a completed slot",
                "user_decision_required": "wait for the quota reset, or explicitly enable a "
                                          "Paid Tier. Code never enables billing."}


def assign_recipe_id(index: int) -> str:
    """Deterministic positional id in the inherited `R-%06d` form."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index > 999_999:
        raise ValueError("recipe index must be an integer in [0, 999999]")
    return f"R-{index:06d}"


def _issue_dicts(issues: Iterable[ValidationIssue]) -> list[dict[str, Any]]:
    return [issue.as_dict() for issue in issues]


class RecipePlanner:
    """Drives provider calls through the validation pipeline.

    C1 exercises the machinery. It does not create a scientific bank: the caller
    supplies the slots, and C2/C3 own the budgets.
    """

    def __init__(self, *, provider: RecipeProvider, config: LLMProviderConfig,
                 ontology: Ontology, sleep: Callable[[float], None] | None = None) -> None:
        self._provider = provider
        self._config = config
        self._ontology = ontology
        self._sleep = sleep if sleep is not None else time.sleep
        self._seen_identities: dict[str, str] = {}
        self.quota = QuotaState(billing_tier=config.quota.billing_tier)

    # --- duplicate detection ------------------------------------------------
    @property
    def seen_identities(self) -> dict[str, str]:
        """canonical recipe identity -> the slot that first produced it."""
        return dict(self._seen_identities)

    def register(self, identity: str, slot_id: str) -> None:
        self._seen_identities.setdefault(identity, slot_id)

    # --- validation ---------------------------------------------------------
    def validate_response(self, raw_text: str, *, slot_id: str,
                          recipes_requested: int,
                          next_recipe_index: int = 0) -> ResponseValidation:
        """Run the whole pipeline over one raw response."""
        response_issues: list[dict[str, Any]] = []

        # stage 1: JSON parsing. Prose, markdown fences or truncation land here.
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return ResponseValidation(
                candidates=[],
                response_issues=[{"stage": "json_parsing", "field": "response",
                                  "reason": f"response is not valid JSON: {exc}"}])

        # stage 2: envelope. Exactly {"recipes": [...]} with the requested count.
        if not isinstance(payload, dict):
            return ResponseValidation([], [{"stage": "envelope_schema", "field": "response",
                                            "reason": "response is not a JSON object"}])
        extra_keys = sorted(set(payload) - {"recipes"})
        if extra_keys:
            response_issues.append({"stage": "envelope_schema", "field": "response",
                                    "reason": f"unknown top-level keys {extra_keys}; only 'recipes' is allowed"})
        if "recipes" not in payload:
            return ResponseValidation([], response_issues + [
                {"stage": "envelope_schema", "field": "recipes", "reason": "missing 'recipes' key"}])
        items = payload["recipes"]
        if not isinstance(items, list):
            return ResponseValidation([], response_issues + [
                {"stage": "envelope_schema", "field": "recipes", "reason": "'recipes' must be an array"}])
        if len(items) != recipes_requested:
            response_issues.append({
                "stage": "envelope_schema", "field": "recipes",
                "reason": f"expected exactly {recipes_requested} recipes, got {len(items)}"})

        candidates: list[RecipeCandidateResult] = []
        for offset, item in enumerate(items):
            candidates.append(self._validate_candidate(
                item, index=offset, slot_id=slot_id, recipe_index=next_recipe_index + offset))
        return ResponseValidation(candidates, response_issues)

    def _validate_candidate(self, item: Any, *, index: int, slot_id: str,
                            recipe_index: int) -> RecipeCandidateResult:
        if not isinstance(item, dict):
            return RecipeCandidateResult(index, CandidateOutcome.REJECTED_ENVELOPE, issues=[
                {"stage": "envelope_schema", "field": f"recipes[{index}]",
                 "reason": "candidate is not a JSON object"}])

        # The model may not supply anything the system owns. This is checked
        # before parsing so the reason is specific rather than "extra key".
        intruding = sorted(set(item) & SYSTEM_OWNED_KEYS)
        if intruding:
            return RecipeCandidateResult(index, CandidateOutcome.REJECTED_SYSTEM_OWNED_FIELD, issues=[
                {"stage": "typed_recipe_schema", "field": f"recipes[{index}]",
                 "reason": f"candidate supplies system-owned fields {intruding}; the system computes "
                           "recipe id, content hash, provenance and validation status"}])

        # `recipe_id` is added, not repaired: the schema requires one and the
        # model was forbidden to provide it.
        payload = {**item, "recipe_id": assign_recipe_id(recipe_index)}

        recipe, issues = validate_payload(
            payload, self._ontology, canonicalize=self._config.allow_ontology_aliases)
        if recipe is None:
            return RecipeCandidateResult(index, CandidateOutcome.REJECTED_SCHEMA,
                                         issues=_issue_dicts(issues))
        if issues:
            return RecipeCandidateResult(index, CandidateOutcome.REJECTED_ONTOLOGY, recipe=recipe,
                                         issues=_issue_dicts(issues))

        text = canonical_json(recipe)
        identity = recipe_hash(recipe)

        # Duplicate detection uses the canonical scientific identity, which is
        # computed over the semantic content only. `recipe_id` is positional, so
        # two candidates with identical content collide here regardless of id.
        content_identity = self.content_identity(recipe)
        first_seen = self._seen_identities.get(content_identity)
        if first_seen is not None:
            return RecipeCandidateResult(
                index, CandidateOutcome.REJECTED_DUPLICATE, recipe=recipe,
                canonical_text=text, recipe_identity=identity,
                issues=[{"stage": "duplicate_detection", "field": f"recipes[{index}]",
                         "reason": f"canonical content duplicates a candidate already accepted "
                                   f"from slot {first_seen!r}"}])
        self._seen_identities[content_identity] = slot_id
        return RecipeCandidateResult(index, CandidateOutcome.ACCEPTED, recipe=recipe,
                                     canonical_text=text, recipe_identity=identity)

    @staticmethod
    def content_identity(recipe: RecipeV11) -> str:
        """Identity over semantic content only, with the positional `recipe_id`
        excluded, so duplicate detection cannot be defeated by renumbering."""
        payload = json.loads(canonical_json(recipe))
        payload.pop("recipe_id", None)
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # --- retry / quota ------------------------------------------------------
    def backoff_seconds(self, attempt: int) -> float:
        retry = self._config.retry
        delay = retry.backoff_initial_seconds * (retry.backoff_multiplier ** max(0, attempt - 1))
        return float(min(delay, retry.backoff_max_seconds))

    def generate_slot(self, request: GenerationRequest, *, recipes_requested: int,
                      next_recipe_index: int = 0,
                      pending_slot_ids: Iterable[str] | None = None
                      ) -> tuple[ProviderGenerationResult, ResponseValidation | None, int]:
        """Attempt one slot under the bounded retry policy.

        Returns the final attempt's result, its validation (if a response was
        produced) and the number of attempts made. Raises `QuotaBlocked` when the
        provider reports daily quota exhaustion.
        """
        retry = self._config.retry
        max_attempts = 1 + retry.semantic_max_retries
        transport_attempts = 0
        attempt = 0
        last: ProviderGenerationResult | None = None
        last_validation: ResponseValidation | None = None

        while attempt < max_attempts:
            attempt += 1
            result = self._provider.generate(request, attempt=attempt)
            last = result

            if result.error is not None:
                error = result.error
                if error.error_class is ErrorClass.QUOTA_EXHAUSTED:
                    self._block_on_quota(request, pending_slot_ids)
                if error.error_class in (ErrorClass.TRANSPORT, ErrorClass.SERVER_ERROR,
                                         ErrorClass.RATE_LIMIT):
                    transport_attempts += 1
                    if transport_attempts >= retry.transport_max_attempts:
                        return result, None, attempt
                    self._sleep(error.retry_after_seconds
                                if error.retry_after_seconds is not None
                                else self.backoff_seconds(transport_attempts))
                    attempt -= 1          # transport failures do not spend the semantic budget
                    continue
                return result, None, attempt      # non-retryable: stop immediately

            validation = self.validate_response(
                result.raw_text or "", slot_id=request.slot_id,
                recipes_requested=recipes_requested, next_recipe_index=next_recipe_index)
            last_validation = validation
            if validation.all_accepted:
                return result, validation, attempt
            # An invalid candidate may be retried under the SAME frozen contract.
            # No field is edited to make it pass.
            if attempt >= max_attempts:
                break
        return last, last_validation, attempt  # type: ignore[return-value]

    def _block_on_quota(self, request: GenerationRequest,
                        pending_slot_ids: Iterable[str] | None) -> None:
        pending = sorted(set(pending_slot_ids or []) | {request.slot_id})
        self.quota.blocked = True
        self.quota.reason = "provider reported daily quota exhaustion (429 quota_exceeded)"
        self.quota.blocked_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.quota.pending_slot_ids = pending
        self.quota.completed_slot_ids = sorted(set(self.quota.completed_slot_ids) - set(pending))
        raise QuotaBlocked(self.quota)

    def mark_slot_completed(self, slot_id: str) -> None:
        if slot_id not in self.quota.completed_slot_ids:
            self.quota.completed_slot_ids.append(slot_id)


def compile_accepted(recipe: RecipeV11, ontology: Ontology, *, bank_id: str) -> Any:
    """Prove a validated recipe reaches the inherited compiler.

    The compiler is the execution authority: a recipe that validates at the LLM
    layer but cannot compile is not acceptable, so C1 tests this boundary rather
    than assuming it.
    """
    from prism_fas.recipes.compile import compile_recipe
    return compile_recipe(recipe, ontology, bank_id=bank_id)


__all__ = ["PIPELINE_STAGES", "SYSTEM_OWNED_KEYS", "CandidateOutcome", "RecipeCandidateResult",
           "ResponseValidation", "QuotaBlocked", "QuotaState", "RecipePlanner",
           "assign_recipe_id", "compile_accepted", "RECIPE_ID_PATTERN"]
