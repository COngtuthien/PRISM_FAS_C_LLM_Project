"""M10 shared contracts: schema versions, statuses, errors and canonical hashing.

Every schema version here is bound into the artifact it names, so a silently
changed format cannot be read as if it were the frozen one. Nothing in this
package may import modal (spec section 12.1 / Table 41), and nothing here may
depend on a target label.

See `docs/M10_EXPERIMENT_CONTRACT.md`, `docs/M10_TARGET_EVALUATION_CONTRACT.md`
and `docs/M10_STATISTICS_CONTRACT.md`.
"""
from __future__ import annotations
import hashlib, json
from typing import Any

M10_MATRIX_SCHEMA_VERSION = "m10-matrix-v1"
M10_REGISTRY_SCHEMA_VERSION = "m10-registry-v1"
M10_PREDICTION_SCHEMA_VERSION = "m10-target-prediction-v1"
M10_PREDICTION_LOCK_SCHEMA_VERSION = "m10-prediction-lock-v1"
M10_SCORING_SCHEMA_VERSION = "m10-scoring-v1"
M10_BOOTSTRAP_SCHEMA_VERSION = "m10-bootstrap-v1"
M10_RELIABILITY_SCHEMA_VERSION = "m10-reliability-v1"
M10_REPORT_SCHEMA_VERSION = "m10-report-v1"

# Terminal statuses a row may end in. A failed row is KEPT with its failure record;
# it is never silently dropped and never becomes a missing row.
TERMINAL_STATUSES = ("COMPLETED", "FAILED", "BLOCKED")
PLANNED_STATUSES = ("PLANNED", "RUNNING")
ALL_STATUSES = PLANNED_STATUSES + TERMINAL_STATUSES

# The M10 stage vocabulary. G1/G2/G5/G6 are M9's, unchanged; G7/G8 are new.
STAGE_ORDER = ("G1", "G2", "G5", "G6", "G7", "G8")

REPLICATION_ROLES = ("spec_mandated", "hypothesis_critical", "diagnostic", "parity", "blocked")
# Only these roles may carry a confidence interval, a significance test or a
# superiority claim (statistics contract section 2).
CLAIM_BEARING_ROLES = frozenset({"spec_mandated", "hypothesis_critical"})

LIVE, SPOOF = 0, 1
DECISIONS = ("live", "spoof", "reject")


class M10ContractError(ValueError):
    """An M10 artifact or config does not satisfy the frozen contract."""


class TargetLabelFirewallViolation(PermissionError):
    """A stage asked for a path or a field its declared permissions forbid."""


class PredictionLockError(RuntimeError):
    """A prediction is unlocked, or its lock disagrees with the frozen artifacts."""


class ScoringRefusal(RuntimeError):
    """G8 refused to score. A refusal is an error, never a fallback path."""


class MetricNotAvailable(ValueError):
    """A metric cannot be computed from the available protocol or population."""


def canonical_json(value: Any) -> str:
    """One canonical text form, used by every identity in this package."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def not_applicable(reason: str, **extra: Any) -> dict[str, Any]:
    """The ONLY way this package reports a metric it cannot compute.

    Never `0.0`, never `null`, never an invented value — a status and the specific
    reason, so a reader can tell "we could not" from "we measured zero".
    """
    if not reason: raise M10ContractError("not_applicable requires a specific reason")
    return {"status": "not_applicable", "reason": str(reason), **extra}


def is_not_applicable(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "not_applicable"


def require_status(status: str) -> str:
    if status not in ALL_STATUSES:
        raise M10ContractError(f"unknown M10 status {status!r}; allowed: {list(ALL_STATUSES)}")
    return status


def require_terminal(status: str) -> str:
    if status not in TERMINAL_STATUSES:
        raise M10ContractError(f"{status!r} is not a terminal status; allowed: {list(TERMINAL_STATUSES)}")
    return status


def require_replication_role(role: str) -> str:
    if role not in REPLICATION_ROLES:
        raise M10ContractError(f"unknown replication role {role!r}; allowed: {list(REPLICATION_ROLES)}")
    return role


def may_carry_statistical_claim(role: str) -> bool:
    return require_replication_role(role) in CLAIM_BEARING_ROLES
