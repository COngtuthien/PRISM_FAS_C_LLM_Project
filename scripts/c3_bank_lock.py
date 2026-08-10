"""Create and verify the immutable C3 generation BANK_LOCK.

    python scripts/c3_bank_lock.py            # create (once) and verify
    python scripts/c3_bank_lock.py --verify   # verify only, never write

This is the gate before any C3 scientific request. It makes no network call and
reads no credential: every identity is re-derived from the code and configuration
on disk and compared with what the user approved. If anything drifted, no lock is
written and the script fails.

The quota snapshot is recorded honestly. AI Studio's RPM/TPM/RPD figures live
behind an authenticated web console this process cannot read, so the snapshot is
marked unavailable with the exact manual step. No number is invented.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from c2c_common import RouteContext, read_json, write_json  # noqa: E402

from prism_fas.llm.bank_lock import (APPROVED_C3_GENERATION_CONTRACT_IDENTITY,  # noqa: E402
                                     BankLockError, build_lock, verify_lock, write_lock_once)

REPORTS = REPO / "reports" / "c3"
DOCS = REPO / "docs" / "c3"
LOCK_PATH = REPORTS / "C3_BANK_LOCK.json"
VERIFICATION_PATH = REPORTS / "C3_BANK_LOCK_VERIFICATION.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                            check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def quota_snapshot() -> dict:
    """What is actually knowable from this process, and what is not."""
    import importlib.metadata as metadata

    try:
        sdk_version = metadata.version("google-genai")
    except Exception:
        sdk_version = None
    return {
        "recorded_at_utc": utc_now(),
        "billing_tier": "free",
        "auto_enable_paid": False,
        "rpm": "NOT_AVAILABLE",
        "tpm": "NOT_AVAILABLE",
        "rpd": "NOT_AVAILABLE",
        "availability": "NOT_PROGRAMMATICALLY_AVAILABLE",
        "reason": "The active project's RPM/TPM/RPD limits are shown only in the AI Studio "
                  "web console, which requires an interactive authenticated session this "
                  "process does not have. No value was invented.",
        "manual_step_before_c3": "Open AI Studio for the project owning GEMINI_API_KEY and "
                                 "record the RPM / TPM / RPD shown for gemini-3.6-flash on the "
                                 "current tier.",
        "measured_free_tier_behaviour_from_earlier_milestones": {
            "source": "reports/c2/C2_RATE_LIMIT_INCIDENTS.json",
            "observation": "roughly 20 requests per rolling multi-minute window for this model "
                           "during the C2 singleton pilot; the C2B and C2C batch requests (1 "
                           "request each) met no rate limit at all",
            "c3_request_count": 12,
            "caveat": "an observation, not a published limit",
        },
        "sdk_version_installed": sdk_version,
        "python_version": platform.python_version(),
    }


def evidence_block() -> dict:
    """Where the approved contract's supporting measurements live."""
    c2c = REPO / "reports" / "c2c"
    payload = {}
    acceptance = c2c / "C2C_ACCEPTANCE.json"
    if acceptance.exists():
        record = read_json(acceptance)
        payload["c2c_acceptance"] = {
            "result": record["result"],
            "artifact": "reports/c2c/C2C_ACCEPTANCE.json",
            "live_batch": {
                "returned_objects": record["live_batch"]["returned_objects"],
                "accepted_objects": record["live_batch"]["accepted_objects"],
                "compiled_objects": record["live_batch"]["compiled_objects"],
                "compiler_failures_among_accepted":
                    record["live_batch"]["compiler_failures_among_accepted"],
            },
            "coverage_axes": record["coverage"]["axes"],
        }
    payload["c2b"] = {
        "result": "BATCH_SHAPE_FAIL",
        "artifact": "reports/c2b/C2B_ACCEPTANCE.json",
        "preserved_unchanged": True,
        "note": "C2B remains historically BATCH_SHAPE_FAIL and was not rewritten.",
    }
    payload["pilot_recipes_entering_c3"] = 0
    payload["c2_c2b_c2c_recipes_entering_c3"] = 0
    return payload


def build_doc(lock: dict, verification: dict) -> str:
    components = lock["components"]
    schedule = lock["scientific_request_schedule"]
    rows = [
        ["provider", components["provider"]],
        ["model", f"`{components['model_id']}`"],
        ["SDK / API surface",
         f"{components['sdk_package']} / {components['api_surface']}"],
        ["thinking", f"thinking_level = {components['thinking_level']}, "
                     f"max_output_tokens = {components['max_output_tokens']}, "
                     f"no sampling controls"],
        ["system prompt identity", f"`{components['system_prompt_identity']}`"],
        ["batch generation-template identity",
         f"`{components['batch_generation_template_identity']}`"],
        ["coverage quota identity", f"`{components['coverage_quota_identity']}`"],
        ["single-recipe schema identity", f"`{components['single_recipe_schema_identity']}`"],
        ["batch-envelope schema identity", f"`{components['batch_envelope_schema_identity']}`"],
        ["ontology identity", f"`{components['ontology_identity']}`"],
        ["route policy identity", f"`{components['route_policy_identity']}`"],
        ["alias policy", f"allow_ontology_aliases = {components['allow_ontology_aliases']}"],
        ["provider config identity", f"`{components['provider_config_identity']}`"],
        ["retry policy", f"`{lock['components']['retry_policy']}`"],
        ["request schedule", f"{schedule['requests']} x {schedule['objects_per_request']} = "
                             f"{schedule['raw_slots']} raw slots; min unique pool "
                             f"{schedule['minimum_unique_pool']}; final bank "
                             f"{schedule['final_bank']}"],
    ]
    table = "\n".join(
        ["| component | frozen value |", "| --- | --- |"]
        + [f"| {name} | {value} |" for name, value in rows])

    checks = "\n".join(
        ["| component | approved | re-derived | matches |", "| --- | --- | --- | --- |"]
        + [f"| {item['component']} | `{item['approved']}` | `{item['actual']}` | "
           f"{'yes' if item['matches'] else '**NO**'} |"
           for item in verification["components_vs_approval"]])

    return f"""# C3 generation BANK_LOCK

**Status: {lock['status']}.** Frozen before any C3 scientific request. This file records
the contract every C3 request must run under; it contains no recipe.

- `bank_lock_identity`: `{lock['bank_lock_identity']}`
- **`C3_GENERATION_CONTRACT_IDENTITY`**: `{lock['composite']['c3_generation_contract_identity']}`

## Frozen components

{table}

## Composite identity

The composite is taken over exactly these keys:

    {', '.join(sorted(lock['composite']['component_keys_in_hash_order']))}

Canonical form: `{lock['composite']['canonical_form']}`

> {lock['composite']['invalidation_rule']}

## Verification against the approval

Every identity below was **re-derived from the code and configuration on disk**, not
copied from the approval text.

{checks}

- lock body hash reproducible: **{verification['bank_lock_identity_in_file'] == verification['bank_lock_identity_recomputed']}**
- composite reproducible from code: **{verification['composite_in_file'] == verification['composite_recomputed']}**
- composite equals the approved value: **{verification['composite_in_file'] == verification['composite_approved']}**
- problems: **{verification['problems'] or 'none'}**

## Route contract

`generator_route` must be exactly `["physics", "gpat"]`. Physics-only and gpat-only are
both rejected; there is no GPAT-only accepted class; silent repair is never permitted.

## Request schedule

{schedule['rule']}

- minimum valid unique pool before selection: **{schedule['minimum_unique_pool']}** —
  {schedule['below_minimum_pool_action']}
- final bank: **{schedule['final_bank']}**
- selection: {schedule['selection']}

## Free-Tier operating policy (approved)

- Free Tier only. Code never enables billing.
- Transient short-window 429: retry the exact frozen request under the approved bounded
  backoff.
- True daily/project quota exhaustion: checkpoint completed scientific requests and stop
  cleanly.
- Quota never changes the provider, model, prompt, schema, ontology, quotas, route policy,
  request schedule or this lock. A resumed run uses the same frozen contract.

## Quota snapshot

{lock['quota_snapshot']['availability']} — {lock['quota_snapshot']['reason']}

Manual step before C3: {lock['quota_snapshot']['manual_step_before_c3']}

## Prohibited during C3

{', '.join(lock['prohibitions_during_c3'])}.

## Immutability

{lock['immutability']['rule']}
"""


def main() -> int:
    verify_only = "--verify" in sys.argv
    context = RouteContext()

    existing = read_json(LOCK_PATH) if LOCK_PATH.exists() else None
    if existing is None and verify_only:
        print("no C3_BANK_LOCK.json to verify")
        return 3

    if existing is None:
        try:
            lock = build_lock(context, generator_code_commit=git("rev-parse", "HEAD"),
                              generated_at_utc=utc_now(), quota_snapshot=quota_snapshot(),
                              evidence=evidence_block())
        except BankLockError as exc:
            print(f"REFUSED to build the BANK_LOCK:\n{exc}")
            return 3
        action = write_lock_once(LOCK_PATH, lock)
        print(f"BANK_LOCK {action}: reports/c3/C3_BANK_LOCK.json")
    else:
        lock = existing
        print("BANK_LOCK already exists; verifying without rewriting")

    verification = verify_lock(lock, context)
    write_json(VERIFICATION_PATH, {
        "schema_version": "c3-bank-lock-verification-v1",
        "milestone": "C3",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "network_calls": 0,
        "credential_read": False,
        "lock_path": "reports/c3/C3_BANK_LOCK.json",
        "approved_composite_identity": APPROVED_C3_GENERATION_CONTRACT_IDENTITY,
        **verification,
    })

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "C3_BANK_LOCK.md").write_text(build_doc(lock, verification), encoding="utf-8")
    print("wrote docs/c3/C3_BANK_LOCK.md")

    print(f"\nbank_lock_identity            {lock['bank_lock_identity']}")
    print(f"C3_GENERATION_CONTRACT_IDENTITY {lock['composite']['c3_generation_contract_identity']}")
    print(f"matches approved value        "
          f"{lock['composite']['c3_generation_contract_identity'] == APPROVED_C3_GENERATION_CONTRACT_IDENTITY}")
    print(f"verified                      {verification['verified']}")
    if verification["problems"]:
        print("problems:", verification["problems"])
    return 0 if verification["verified"] else 3


if __name__ == "__main__":
    sys.exit(main())
