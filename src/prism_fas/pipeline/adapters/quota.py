"""The C3 live quota snapshot: schema, validation and template.

The provider exposes no programmatic quota endpoint for this tier, so the
remaining-request figure the C3 live preflight needs can only come from a human
reading the AI Studio dashboard. That makes the snapshot a piece of *operational
provenance*: it records what a person observed, when, and how.

Everything here follows from that one fact.

**Nothing contacts Google.** This module builds and checks a document. It has no
network path, by construction.

**UNKNOWN is a first-class value.** The dashboard frequently shows "No data
available" for current usage, and the honest recording of that is
``current_remaining_rpd: "UNKNOWN"``. A schema that forced an integer would be a
schema that forced a guess, and a guessed remaining-quota number is worse than
none: it would be used to decide whether 12 requests fit.

**An invented observation is refused.** A snapshot must declare where its values
came from. `observed_by: "inferred"` and a numeric remaining count together are
rejected, because that combination is exactly what fabrication looks like.

**It is not a tuning signal.** Quota is operational configuration under L.12 and
is never a treatment factor. It may stop a run or delay it; it may never change
a prompt, a schedule, a selection rule or a bank.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterError
from prism_fas.pipeline.state import atomic_write_json

QUOTA_SNAPSHOT_SCHEMA_VERSION = "c3-quota-snapshot-v1"

#: The literal that means "the dashboard did not tell us". Never an int.
UNKNOWN = "UNKNOWN"

#: How a value reached the snapshot. `user_observed` is the only provenance that
#: may accompany a concrete remaining-quota number.
OBSERVATION_SOURCES: tuple[str, ...] = ("user_observed", "not_available", "inferred")

#: Fields the user reads off the dashboard. Present as a schema so the later
#: materialization step has something to fill rather than something to invent.
REQUIRED_FIELDS: tuple[str, ...] = (
    "project", "tier", "model", "rpm_limit", "tpm_limit", "rpd_limit",
    "observation_window", "usage_dashboard", "current_remaining_rpd",
    "observed_by", "observed_at_utc")

RELATIVE_PATH = Path("reports/c3/live/C3_QUOTA_SNAPSHOT.json")
TEMPLATE_RELATIVE_PATH = Path("reports/c3/live/C3_QUOTA_SNAPSHOT_TEMPLATE.json")


class QuotaSnapshotError(AdapterError):
    """The snapshot is missing, malformed, or claims more than it observed."""


@dataclass(frozen=True)
class QuotaSnapshot:
    """One recorded observation of the provider's quota dashboard."""

    project: str
    tier: str
    model: str
    rpm_limit: int | str
    tpm_limit: int | str
    rpd_limit: int | str
    observation_window: str
    usage_dashboard: str
    current_remaining_rpd: int | str
    observed_by: str
    observed_at_utc: str
    materialized: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def remaining_is_known(self) -> bool:
        return isinstance(self.current_remaining_rpd, int)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUOTA_SNAPSHOT_SCHEMA_VERSION,
            "artifact_kind": "OPERATIONAL_PROVENANCE",
            "is_scientific_tuning_signal": False,
            "project": self.project,
            "tier": self.tier,
            "model": self.model,
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "rpd_limit": self.rpd_limit,
            "observation_window": self.observation_window,
            "usage_dashboard": self.usage_dashboard,
            "current_remaining_rpd": self.current_remaining_rpd,
            "remaining_is_known": self.remaining_is_known,
            "observed_by": self.observed_by,
            "observed_at_utc": self.observed_at_utc,
            "materialized": self.materialized,
            "notes": list(self.notes),
            "authority": (
                "operational provenance under L.12. It may delay or stop a run; it may "
                "never change a prompt, schedule, selection rule, quota contract or bank."),
        }


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Return every problem with a snapshot document. Empty means usable."""
    problems: list[str] = []

    if payload.get("schema_version") != QUOTA_SNAPSHOT_SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {QUOTA_SNAPSHOT_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}")

    for name in REQUIRED_FIELDS:
        if name not in payload:
            problems.append(f"missing required field {name!r}")

    remaining = payload.get("current_remaining_rpd")
    if remaining is not None:
        if isinstance(remaining, bool) or not isinstance(remaining, (int, str)):
            problems.append("current_remaining_rpd must be an integer or the literal 'UNKNOWN'")
        elif isinstance(remaining, str) and remaining != UNKNOWN:
            problems.append(
                f"current_remaining_rpd must be an integer or {UNKNOWN!r}, got {remaining!r}")
        elif isinstance(remaining, int) and remaining < 0:
            problems.append("current_remaining_rpd cannot be negative")

    source = payload.get("observed_by")
    if source is not None and source not in OBSERVATION_SOURCES:
        problems.append(f"observed_by must be one of {OBSERVATION_SOURCES}, got {source!r}")

    # The rule that makes the schema worth having: a concrete remaining count is
    # only meaningful if a human actually read it off the dashboard.
    if isinstance(remaining, int) and source != "user_observed":
        problems.append(
            f"current_remaining_rpd is the concrete value {remaining} but observed_by is "
            f"{source!r}; a remaining-quota number may only accompany 'user_observed'. "
            f"Record {UNKNOWN!r} instead of inferring it.")

    if payload.get("materialized") is True and not payload.get("observed_at_utc"):
        problems.append("a materialized snapshot must carry the UTC time it was observed")

    if payload.get("is_scientific_tuning_signal") is True:
        problems.append("a quota snapshot is operational provenance, never a tuning signal")

    return problems


def load(repo: Path, *, relative: Path = RELATIVE_PATH) -> QuotaSnapshot:
    """Load and validate the snapshot, or explain precisely what is wrong."""
    import json

    path = Path(repo) / relative
    if not path.exists():
        raise QuotaSnapshotError(
            f"{relative.as_posix()} does not exist. The C3 live preflight needs a quota "
            "snapshot, and it can only be produced by a human reading the AI Studio "
            "dashboard; nothing may generate it automatically.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QuotaSnapshotError(f"{relative.as_posix()} is not readable JSON ({error})") from error

    problems = validate_payload(payload)
    if problems:
        raise QuotaSnapshotError(
            f"{relative.as_posix()} is not a valid quota snapshot: " + "; ".join(problems))

    return QuotaSnapshot(
        project=payload["project"], tier=payload["tier"], model=payload["model"],
        rpm_limit=payload["rpm_limit"], tpm_limit=payload["tpm_limit"],
        rpd_limit=payload["rpd_limit"],
        observation_window=payload["observation_window"],
        usage_dashboard=payload["usage_dashboard"],
        current_remaining_rpd=payload["current_remaining_rpd"],
        observed_by=payload["observed_by"], observed_at_utc=payload["observed_at_utc"],
        materialized=bool(payload.get("materialized", False)),
        notes=list(payload.get("notes", [])))


def build_template(*, model: str) -> dict[str, Any]:
    """An unmaterialized template, safe to commit.

    Every observed value is `UNKNOWN` and `materialized` is false, so the
    template cannot be mistaken for evidence. `observed_at_utc` is empty rather
    than "now": stamping a time here would date an observation nobody made,
    which is the specific dishonesty this file exists to prevent.
    """
    return {
        "schema_version": QUOTA_SNAPSHOT_SCHEMA_VERSION,
        "artifact_kind": "OPERATIONAL_PROVENANCE_TEMPLATE",
        "is_scientific_tuning_signal": False,
        "materialized": False,
        "project": UNKNOWN,
        "tier": UNKNOWN,
        "model": model,
        "rpm_limit": UNKNOWN,
        "tpm_limit": UNKNOWN,
        "rpd_limit": UNKNOWN,
        "observation_window": UNKNOWN,
        "usage_dashboard": UNKNOWN,
        "current_remaining_rpd": UNKNOWN,
        "observed_by": "not_available",
        "observed_at_utc": "",
        "notes": [
            "TEMPLATE. Not an observation. The C3 live preflight rejects it until a human "
            "fills it in from the AI Studio dashboard and sets materialized=true.",
            "current_remaining_rpd stays 'UNKNOWN' unless the dashboard actually shows a "
            "number. Do not compute it from the request count; an inferred value is refused.",
            "No field here may influence a prompt, schedule, selection rule or bank.",
        ],
        "how_to_materialize": [
            "1. Open AI Studio for the project that owns the API key.",
            "2. Read the tier, model, RPM/TPM/RPD limits and the usage panel verbatim.",
            "3. Copy them in, set observed_by='user_observed' and observed_at_utc to the "
            "UTC time you read them, and set materialized=true.",
            "4. If the usage panel shows no data, leave current_remaining_rpd as 'UNKNOWN'.",
        ],
    }


def write_template(repo: Path, *, model: str,
                   relative: Path = TEMPLATE_RELATIVE_PATH) -> Path:
    path = Path(repo) / relative
    atomic_write_json(path, build_template(model=model))
    return path


def preflight(repo: Path, *, required: bool) -> dict[str, Any]:
    """Report whether a usable snapshot exists, without ever creating one.

    `required=False` is the offline case: the check runs, reports what it found
    and does not fail the stage. `required=True` is live generation, where a
    missing or invalid snapshot is a hard stop.
    """
    try:
        snapshot = load(repo)
    except QuotaSnapshotError as error:
        return {
            "ok": not required,
            "present": (Path(repo) / RELATIVE_PATH).exists(),
            "usable": False,
            "required": required,
            "problem": str(error),
            "summary": "no usable quota snapshot; live generation stays gated" if required
                       else "no quota snapshot yet, which is expected before live generation",
        }
    return {
        "ok": True,
        "present": True,
        "usable": True,
        "required": required,
        "materialized": snapshot.materialized,
        "remaining_is_known": snapshot.remaining_is_known,
        "current_remaining_rpd": snapshot.current_remaining_rpd,
        "observed_by": snapshot.observed_by,
        "observed_at_utc": snapshot.observed_at_utc,
        "summary": "a valid quota snapshot is present",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = ["QUOTA_SNAPSHOT_SCHEMA_VERSION", "UNKNOWN", "OBSERVATION_SOURCES",
           "REQUIRED_FIELDS", "RELATIVE_PATH", "TEMPLATE_RELATIVE_PATH",
           "QuotaSnapshotError", "QuotaSnapshot", "validate_payload", "load",
           "build_template", "write_template", "preflight", "utc_now"]
