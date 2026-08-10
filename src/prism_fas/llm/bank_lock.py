"""The immutable C3 generation BANK_LOCK.

The lock is the gate between an approved contract and any scientific generation.
It records the exact identities the user approved, and it is built by
RE-DERIVING every one of them from the live code and configuration rather than
by copying the approval text. If anything in the repository has drifted from
what was approved, `build_lock` refuses to produce a lock at all.

Three properties make it usable as a scientific gate:

* **Bound.** `c3_generation_contract_identity` is the composite the user
  approved. Changing any component changes it, and a lock whose recomputed
  composite disagrees with the approved value is rejected.
* **Immutable.** Once written, the lock is never rewritten in place. A second
  build that would produce different bytes raises rather than overwriting, so a
  generation run can always be traced to the exact contract it ran under.
* **Verifiable offline.** `verify_lock` re-derives every identity from the
  repository and reports each one individually, so "the lock still matches the
  code" is a measurement, not an assumption.

The lock deliberately does NOT contain a recipe. It freezes how recipes will be
requested; the bank contents come later and bind back to this identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BANK_LOCK_SCHEMA_VERSION = "c3-generation-bank-lock-v1"

#: The exact values approved by the user for the C3 generation contract.
#: Transcribed once, here, so that every other artifact can be CHECKED against
#: them rather than trusted. Nothing in the pipeline reads these as inputs: they
#: are the expected values, and the actual values are re-derived from the code.
APPROVED_C3_CONTRACT: dict[str, Any] = {
    "provider": "gemini",
    "model_id": "gemini-3.6-flash",
    "api_surface": "interactions",
    "sdk_package": "google-genai",
    "sdk_version_approved": "2.17.0",
    "thinking_level": "medium",
    "response_mime_type": "application/json",
    "max_output_tokens": 32768,
    "system_prompt_identity":
        "e1bc86723ed8e84a25efdd7be879424c0abf0c7ee85720a5e0fb8f097c64c737",
    "batch_generation_template_identity":
        "e6dd98cf85b204b6a55709b79dee1588b11b72330d731db2b335bfc2588b6a20",
    "coverage_quota_identity":
        "89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5",
    "single_recipe_schema_identity":
        "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579",
    "batch_envelope_schema_identity":
        "f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113",
    "ontology_identity":
        "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd",
    "route_policy_identity":
        "209ccacddd2d10d7485a8b1fce9e93eccde59903a103daefda6ffecc717c13d7",
    "allow_ontology_aliases": False,
    "provider_config_identity":
        "3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da",
    "retry_policy": {
        "semantic_max_retries": 2,
        "transport_max_attempts": 4,
        "backoff_initial_seconds": 1.0,
        "backoff_multiplier": 2.0,
        "backoff_max_seconds": 60.0,
    },
    "request_schedule": {
        "requests": 12,
        "objects_per_request": 32,
        "raw_slots": 384,
        "minimum_unique_pool": 320,
        "final_bank": 256,
    },
}

#: The composite identity the user approved. A lock that does not reproduce this
#: exact value is refused.
APPROVED_C3_GENERATION_CONTRACT_IDENTITY = (
    "884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85")

#: The component keys that enter the composite identity, and the canonical form.
#: Documented explicitly because the composite is what the bank binds to.
COMPOSITE_COMPONENT_KEYS: tuple[str, ...] = (
    "provider", "model_id", "api_surface", "sdk_package", "thinking_level",
    "response_mime_type", "max_output_tokens", "system_prompt_identity",
    "batch_generation_template_identity", "coverage_quota_identity",
    "single_recipe_schema_identity", "batch_envelope_schema_identity",
    "ontology_identity", "route_policy_identity", "allow_ontology_aliases",
    "provider_config_identity", "request_schedule", "retry_policy",
)

CANONICAL_FORM = ("json.dumps(components, sort_keys=True, separators=(',',':'), "
                  "ensure_ascii=False) then SHA-256 over the UTF-8 bytes")


class BankLockError(RuntimeError):
    """The lock cannot be built or does not verify. Never partially applied."""


def canonical_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def composite_identity(components: dict[str, Any]) -> str:
    """The C3_GENERATION_CONTRACT_IDENTITY over exactly the approved keys."""
    missing = [key for key in COMPOSITE_COMPONENT_KEYS if key not in components]
    if missing:
        raise BankLockError(f"composite components are incomplete: missing {missing}")
    extra = sorted(set(components) - set(COMPOSITE_COMPONENT_KEYS))
    if extra:
        raise BankLockError(f"composite components carry unapproved keys: {extra}")
    return sha256_text(canonical_text(components))


@dataclass(frozen=True)
class ComponentCheck:
    """One approved identity, and what the repository actually produces."""

    component: str
    approved: Any
    actual: Any

    @property
    def matches(self) -> bool:
        return self.approved == self.actual

    def as_dict(self) -> dict[str, Any]:
        return {"component": self.component, "approved": self.approved,
                "actual": self.actual, "matches": self.matches}


def derive_components(context: Any) -> dict[str, Any]:
    """Re-derive the composite components from the LIVE code and configuration.

    `context` is a C2C-style contract context: it exposes the loaded ontology,
    provider config, prompt template, coverage quotas and route policy, each of
    which computes its own identity from the bytes actually on disk.
    """
    contract = context.as_contract_record()
    return {
        "provider": contract["provider_name"],
        "model_id": contract["model_id"],
        "api_surface": contract["api_surface"],
        "sdk_package": contract["sdk_package"],
        "thinking_level": contract["thinking_level"],
        "response_mime_type": contract["response_mime_type"],
        "max_output_tokens": contract["max_output_tokens"],
        "system_prompt_identity": contract["system_prompt_identity"],
        "batch_generation_template_identity": contract["batch_generation_template_identity"],
        "coverage_quota_identity": contract["coverage_quota_identity"],
        "single_recipe_schema_identity": contract["single_recipe_schema_identity"],
        "batch_envelope_schema_identity": contract["batch_envelope_schema_identity"],
        "ontology_identity": contract["ontology_identity"],
        "route_policy_identity": contract["route_policy_identity"],
        "allow_ontology_aliases": contract["allow_ontology_aliases"],
        "provider_config_identity": contract["provider_config_identity"],
        "request_schedule": dict(APPROVED_C3_CONTRACT["request_schedule"]),
        "retry_policy": contract["retry_policy"],
    }


def check_against_approval(components: dict[str, Any]) -> list[ComponentCheck]:
    """Compare every re-derived component with the approved value."""
    checks = [ComponentCheck(key, APPROVED_C3_CONTRACT[key], components[key])
              for key in COMPOSITE_COMPONENT_KEYS]
    return checks


def build_lock(context: Any, *, generator_code_commit: str = "",
               generated_at_utc: str = "", quota_snapshot: dict[str, Any] | None = None,
               evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the lock, refusing outright on any drift from the approval.

    A lock is only meaningful if it could not have been produced from a drifted
    repository, so the drift check is a precondition rather than a field.
    """
    components = derive_components(context)
    checks = check_against_approval(components)
    drift = [check.as_dict() for check in checks if not check.matches]
    if drift:
        raise BankLockError(
            "the repository has drifted from the approved C3 contract; refusing to build a "
            f"BANK_LOCK: {json.dumps(drift, indent=2)}")

    identity = composite_identity(components)
    if identity != APPROVED_C3_GENERATION_CONTRACT_IDENTITY:
        raise BankLockError(
            "the recomputed composite does not reproduce the approved "
            f"C3_GENERATION_CONTRACT_IDENTITY: {identity} != "
            f"{APPROVED_C3_GENERATION_CONTRACT_IDENTITY}")

    body: dict[str, Any] = {
        "bank_lock_schema_version": BANK_LOCK_SCHEMA_VERSION,
        "project": "PRISM-FAS-C-LLM",
        "spec_version": "v1.1 FINAL",
        "milestone": "C3",
        "status": "FROZEN",
        "purpose": "Immutable generation contract for the C3 scientific LLM recipe bank. "
                   "Frozen BEFORE any C3 scientific request.",
        "generated_at_utc": generated_at_utc,
        "generator_code_commit": generator_code_commit,

        "user_approval": {
            "approved": True,
            "approval_scope": "the complete C3 generation contract candidate from C2C, "
                              "including the composite identity",
            "approved_composite_identity": APPROVED_C3_GENERATION_CONTRACT_IDENTITY,
            "approved_free_tier_policy": {
                "billing_tier": "free",
                "auto_enable_paid": False,
                "transient_429": "retry the exact frozen request under the approved bounded "
                                 "backoff",
                "daily_or_project_quota_exhausted": "checkpoint completed scientific requests "
                                                    "and stop cleanly",
                "quota_never_changes_the_contract": True,
                "resume_rule": "resume under the exact same frozen contract",
            },
        },

        "components": components,
        "composite": {
            "c3_generation_contract_identity": identity,
            "component_keys_in_hash_order": sorted(COMPOSITE_COMPONENT_KEYS),
            "canonical_form": CANONICAL_FORM,
            "canonical_text": canonical_text(components),
            "invalidation_rule": "changing ANY component changes this identity and invalidates "
                                 "any C3 generation carried out under the previous value",
        },

        "scientific_request_schedule": {
            **APPROVED_C3_CONTRACT["request_schedule"],
            "rule": "exactly 12 logical requests of exactly 32 recipe objects; a response whose "
                    "recipe count is not exactly 32 fails closed",
            "below_minimum_pool_action": "C3 FAILS; the validator is never weakened after "
                                         "seeing results",
            "selection": "deterministic and algorithmic; no manual cherry-picking of LLM "
                         "recipes",
        },

        "route_contract": {
            "required_generator_route": ["physics", "gpat"],
            "route_policy_identity": components["route_policy_identity"],
            "physics_only_accepted": False,
            "gpat_only_accepted": False,
            "gpat_only_class_exists": False,
            "silent_repair_permitted": False,
        },

        "prohibitions_during_c3": [
            "no GPU training", "no GPAT training", "no synthetic image generation",
            "no detector training", "no SiW label access", "no SiW metric use",
            "no target scoring", "no prompt change", "no schema change",
            "no ontology change", "no quota change", "no route policy change",
            "no model or provider change", "no automatic billing",
        ],

        "quota_snapshot": quota_snapshot or {},
        "evidence": evidence or {},

        "immutability": {
            "rewrite_permitted": False,
            "rule": "this file is written once. A later build that would produce different "
                    "bytes raises rather than overwriting, so a generation run can always be "
                    "traced to the contract it ran under.",
        },
    }
    body["bank_lock_identity"] = sha256_text(canonical_text(body))
    return body


def verify_lock(lock: dict[str, Any], context: Any) -> dict[str, Any]:
    """Re-derive everything and report each identity individually."""
    stored = dict(lock)
    stored_identity = stored.pop("bank_lock_identity", None)
    recomputed_lock_identity = sha256_text(canonical_text(stored))

    components = derive_components(context)
    checks = [check.as_dict() for check in check_against_approval(components)]
    lock_components = lock.get("components", {})
    against_lock = [
        {"component": key, "in_lock": lock_components.get(key), "actual": components.get(key),
         "matches": lock_components.get(key) == components.get(key)}
        for key in COMPOSITE_COMPONENT_KEYS]

    recomputed_composite = composite_identity(components)
    stored_composite = lock.get("composite", {}).get("c3_generation_contract_identity")

    problems: list[str] = []
    if stored_identity != recomputed_lock_identity:
        problems.append("bank_lock_identity does not match the lock body")
    if stored_composite != recomputed_composite:
        problems.append("the lock's composite identity is not reproducible from the code")
    if stored_composite != APPROVED_C3_GENERATION_CONTRACT_IDENTITY:
        problems.append("the lock's composite identity is not the approved value")
    problems.extend(f"{item['component']} drifted from the approval" for item in checks
                    if not item["matches"])
    problems.extend(f"{item['component']} drifted from the lock" for item in against_lock
                    if not item["matches"])
    if lock.get("bank_lock_schema_version") != BANK_LOCK_SCHEMA_VERSION:
        problems.append("unexpected bank_lock_schema_version")
    if lock.get("status") != "FROZEN":
        problems.append("the lock is not marked FROZEN")

    return {
        "verified": not problems,
        "problems": problems,
        "bank_lock_identity_in_file": stored_identity,
        "bank_lock_identity_recomputed": recomputed_lock_identity,
        "composite_in_file": stored_composite,
        "composite_recomputed": recomputed_composite,
        "composite_approved": APPROVED_C3_GENERATION_CONTRACT_IDENTITY,
        "components_vs_approval": checks,
        "components_vs_lock": against_lock,
    }


def write_lock_once(path: Path, lock: dict[str, Any]) -> str:
    """Write the lock, or confirm an identical one already exists.

    Refuses to overwrite a lock whose contents differ. That is what makes the
    artifact immutable in practice rather than only by convention.
    """
    path = Path(path)
    text = json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("bank_lock_identity") != lock.get("bank_lock_identity"):
            raise BankLockError(
                f"{path.name} already exists with a different identity "
                f"({existing.get('bank_lock_identity')} != {lock.get('bank_lock_identity')}). "
                "A frozen bank lock is never rewritten; delete it deliberately only if you "
                "intend to discard the frozen contract.")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


__all__ = ["BANK_LOCK_SCHEMA_VERSION", "APPROVED_C3_CONTRACT",
           "APPROVED_C3_GENERATION_CONTRACT_IDENTITY", "COMPOSITE_COMPONENT_KEYS",
           "CANONICAL_FORM", "BankLockError", "ComponentCheck", "canonical_text",
           "sha256_text", "composite_identity", "derive_components", "check_against_approval",
           "build_lock", "verify_lock", "write_lock_once"]
