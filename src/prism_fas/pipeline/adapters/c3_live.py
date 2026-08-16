"""The C3 live generation state machine: 12 logical requests, 32 slots each.

This module owns *when* a request runs and *whether* it may run again. It does
not own how a request is built, validated, retried or scored — `RouteContext`
builds it and `RecipePlanner.generate_slot` executes it under the frozen retry
policy. Keeping that line sharp is what makes the state machine safe to change
without touching science.

The frozen schedule (§7.8, §7.8.1) is 12 logical requests of exactly 32 recipe
objects, giving 384 raw slots for the LLM arm. Those numbers are read from the
canonical module, never restated here, and the plan is checked against them
before anything executes.

Why a persistent per-request record rather than a counter: the failure this
guards against is a crash between "the provider answered" and "the answer was
archived". A counter cannot distinguish that from "the request never ran", and
guessing wrong either loses 32 paid-for candidates or spends a second request on
slots that already exist. So each of the 12 requests carries its own status, its
own attempt count, its own slot range and its own archive identity, written
atomically after every transition.

Two rules have teeth:

* **COMPLETED_VALID is terminal.** A completed request is never re-issued, not
  by a restart, not by `--resume`, not by a second invocation. L.11 states this
  for the C3 archive specifically, and it is the difference between a resumable
  run and one that quietly re-spends a free-tier daily quota.
* **A drifted archive fails closed.** If a completed request's archived response
  no longer hashes to what the record says, the run stops. It does not
  regenerate: a silent regeneration would replace evidence that something is
  wrong with evidence that looks fine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from prism_fas.llm.contracts import ErrorClass, GenerationRequest, ProviderGenerationResult
from prism_fas.pipeline.adapters import AdapterError
from prism_fas.pipeline.state import atomic_write_json

LIVE_STATE_SCHEMA_VERSION = "c3-live-generation-state-v1"


class RequestStatus(str, Enum):
    """The lifecycle of one logical request. Closed vocabulary."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_VALID = "COMPLETED_VALID"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_BLOCKING = "FAILED_BLOCKING"


#: A run may continue past these; anything else stops the run.
RESUMABLE = frozenset({RequestStatus.NOT_STARTED, RequestStatus.IN_PROGRESS,
                       RequestStatus.FAILED_RETRYABLE})

#: Provider error classes that leave the request retryable on a later run.
#: RATE_LIMIT and QUOTA_EXHAUSTED are separated deliberately: a rate limit is a
#: pause, a daily quota is a stop, and both must resume to the SAME request.
_RETRYABLE_ERRORS = frozenset({ErrorClass.TRANSPORT, ErrorClass.SERVER_ERROR,
                               ErrorClass.RATE_LIMIT, ErrorClass.INVALID_CANDIDATE})
_BLOCKING_STOP_ERRORS = frozenset({ErrorClass.QUOTA_EXHAUSTED})


class LiveStateError(AdapterError):
    """The live state is unusable, inconsistent, or would be violated."""


class CompletedRequestDrift(LiveStateError):
    """A completed request's archive no longer matches its record."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class LogicalRequestRecord:
    """One of the 12 logical requests, and everything needed to resume it."""

    logical_request_index: int
    logical_request_id: str
    slot_start: int
    slot_end: int
    request_identity: str = ""
    attempt_count: int = 0
    status: str = RequestStatus.NOT_STARTED.value
    archive_identity: str | None = None
    raw_response_sha256: str | None = None
    accepted_recipe_count: int | None = None
    last_provider_classification: str | None = None
    next_permitted_action: str = "GENERATE"
    first_started_utc: str | None = None
    last_transition_utc: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def slot_count(self) -> int:
        return self.slot_end - self.slot_start + 1

    @property
    def state(self) -> RequestStatus:
        return RequestStatus(self.status)

    @property
    def complete(self) -> bool:
        return self.state is RequestStatus.COMPLETED_VALID

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plan(*, requests: int, objects_per_request: int, raw_slots: int,
               logical_id_prefix: str = "c3-llm-req") -> list[LogicalRequestRecord]:
    """The frozen 12x32 plan, refusing any arithmetic that is not the contract.

    The guard is not defensive padding. The single most damaging silent error
    available here is a plan of the wrong size — 36 requests instead of 12, or 31
    slots instead of 32 — and both would look completely normal in a log.
    """
    if requests * objects_per_request != raw_slots:
        raise LiveStateError(
            f"the frozen schedule does not multiply out: {requests} requests x "
            f"{objects_per_request} objects != {raw_slots} raw slots")
    plan: list[LogicalRequestRecord] = []
    for index in range(requests):
        start = index * objects_per_request
        plan.append(LogicalRequestRecord(
            logical_request_index=index,
            logical_request_id=f"{logical_id_prefix}-{index + 1:02d}",
            slot_start=start,
            slot_end=start + objects_per_request - 1))
    return plan


@dataclass
class LiveGenerationState:
    """The persistent state of one C3 live generation run."""

    path: Path
    arm: str
    requests: list[LogicalRequestRecord]
    objects_per_request: int
    raw_slots: int
    generation_contract_identity: str = ""
    bank_contract_identity: str = ""
    provider_binding: str = "none"
    execution_profile: str = ""
    scientific_eligible: bool = False
    created_utc: str = field(default_factory=_utc)
    updated_utc: str = field(default_factory=_utc)
    provider_calls_total: int = 0
    blocked_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    # --- derived views ------------------------------------------------------

    @property
    def completed(self) -> list[LogicalRequestRecord]:
        return [item for item in self.requests if item.complete]

    @property
    def all_complete(self) -> bool:
        return len(self.completed) == len(self.requests)

    @property
    def next_request(self) -> LogicalRequestRecord | None:
        """The first request that still needs work, in plan order.

        Plan order, not "first incomplete after the last completed": a resume
        must land on request 6 whether the run died before it started or during
        it, and both look identical from the far end of the list.
        """
        for item in self.requests:
            if item.state in RESUMABLE:
                return item
            if item.state is RequestStatus.FAILED_BLOCKING:
                return item
        return None

    @property
    def resume_cursor(self) -> dict[str, Any]:
        """Exactly where a resume would restart. Serialized into PIPELINE_STATE."""
        nxt = self.next_request
        return {
            "arm": self.arm,
            "all_complete": self.all_complete,
            "completed_requests": len(self.completed),
            "total_requests": len(self.requests),
            "next_logical_request_index": nxt.logical_request_index if nxt else None,
            "next_logical_request_id": nxt.logical_request_id if nxt else None,
            "next_slot_start": nxt.slot_start if nxt else None,
            "next_slot_end": nxt.slot_end if nxt else None,
            "next_permitted_action": nxt.next_permitted_action if nxt else "FINALIZE",
            "blocked_reason": self.blocked_reason,
        }

    def counts(self) -> dict[str, int]:
        tally = {status.value: 0 for status in RequestStatus}
        for item in self.requests:
            tally[item.status] += 1
        return tally

    # --- persistence --------------------------------------------------------

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_STATE_SCHEMA_VERSION,
            "arm": self.arm,
            "execution_profile": self.execution_profile,
            "scientific_eligible": self.scientific_eligible,
            "provider_binding": self.provider_binding,
            "objects_per_request": self.objects_per_request,
            "raw_slots": self.raw_slots,
            "logical_request_count": len(self.requests),
            "generation_contract_identity": self.generation_contract_identity,
            "bank_contract_identity": self.bank_contract_identity,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "provider_calls_total": self.provider_calls_total,
            "blocked_reason": self.blocked_reason,
            "status_counts": self.counts(),
            "resume_cursor": self.resume_cursor,
            "requests": [item.as_dict() for item in self.requests],
            "history": list(self.history),
        }

    def save(self) -> None:
        self.updated_utc = _utc()
        atomic_write_json(self.path, self.as_payload())

    @classmethod
    def load(cls, path: Path) -> "LiveGenerationState":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise LiveStateError(
                f"{path} is not readable JSON; refusing to resume a live generation run "
                f"from an ambiguous state ({error})") from error
        if payload.get("schema_version") != LIVE_STATE_SCHEMA_VERSION:
            raise LiveStateError(
                f"{path} has schema_version {payload.get('schema_version')!r}, expected "
                f"{LIVE_STATE_SCHEMA_VERSION!r}")
        requests = [LogicalRequestRecord(**item) for item in payload["requests"]]
        return cls(
            path=Path(path), arm=payload["arm"], requests=requests,
            objects_per_request=payload["objects_per_request"],
            raw_slots=payload["raw_slots"],
            generation_contract_identity=payload.get("generation_contract_identity", ""),
            bank_contract_identity=payload.get("bank_contract_identity", ""),
            provider_binding=payload.get("provider_binding", "none"),
            execution_profile=payload.get("execution_profile", ""),
            scientific_eligible=bool(payload.get("scientific_eligible", False)),
            created_utc=payload.get("created_utc", _utc()),
            updated_utc=payload.get("updated_utc", _utc()),
            provider_calls_total=int(payload.get("provider_calls_total", 0)),
            blocked_reason=payload.get("blocked_reason"),
            history=list(payload.get("history", [])))

    @classmethod
    def open(cls, path: Path, *, arm: str, schedule: dict[str, int],
             generation_contract_identity: str = "", bank_contract_identity: str = "",
             execution_profile: str = "", provider_binding: str = "none",
             resume: bool = True) -> "LiveGenerationState":
        """Load an existing run or start a new one.

        A resume against an existing file keeps that file's records untouched.
        Starting fresh where completed work exists is refused, because that is
        how 12 paid-for requests become 24.
        """
        path = Path(path)
        if path.exists():
            state = cls.load(path)
            if not resume and state.completed:
                raise LiveStateError(
                    f"{path.name} already records {len(state.completed)} completed logical "
                    "request(s). Starting a fresh run would re-issue them; pass resume=True "
                    "to continue, or move the existing state aside deliberately.")
            state.provider_binding = provider_binding or state.provider_binding
            state.execution_profile = execution_profile or state.execution_profile
            return state
        return cls(
            path=path, arm=arm,
            requests=build_plan(requests=schedule["requests"],
                                objects_per_request=schedule["objects_per_request"],
                                raw_slots=schedule["raw_slots"]),
            objects_per_request=schedule["objects_per_request"],
            raw_slots=schedule["raw_slots"],
            generation_contract_identity=generation_contract_identity,
            bank_contract_identity=bank_contract_identity,
            execution_profile=execution_profile,
            provider_binding=provider_binding)

    # --- transitions --------------------------------------------------------

    def _transition(self, record: LogicalRequestRecord, status: RequestStatus, *,
                    note: str, action: str) -> None:
        record.status = status.value
        record.next_permitted_action = action
        record.last_transition_utc = _utc()
        if note:
            record.notes.append(note)
        self.history.append({
            "utc": record.last_transition_utc,
            "logical_request_id": record.logical_request_id,
            "status": status.value,
            "attempt_count": record.attempt_count,
            "note": note,
        })
        self.save()

    def mark_in_progress(self, record: LogicalRequestRecord, *,
                         request_identity: str) -> None:
        record.attempt_count += 1
        record.request_identity = request_identity
        if record.first_started_utc is None:
            record.first_started_utc = _utc()
        self._transition(record, RequestStatus.IN_PROGRESS,
                         note=f"attempt {record.attempt_count} started", action="AWAIT_RESULT")

    def mark_completed(self, record: LogicalRequestRecord, *, archive_identity: str,
                       raw_response_sha256: str, accepted_recipe_count: int) -> None:
        if accepted_recipe_count != self.objects_per_request:
            raise LiveStateError(
                f"{record.logical_request_id} accepted {accepted_recipe_count} recipes but the "
                f"frozen contract requires exactly {self.objects_per_request}; a batch of the "
                "wrong size is a validation failure and is never completed")
        record.archive_identity = archive_identity
        record.raw_response_sha256 = raw_response_sha256
        record.accepted_recipe_count = accepted_recipe_count
        record.last_provider_classification = "OK"
        self._transition(record, RequestStatus.COMPLETED_VALID,
                         note="all 32 candidates accepted", action="NONE")

    def mark_retryable(self, record: LogicalRequestRecord, *, classification: str,
                       note: str = "") -> None:
        record.last_provider_classification = classification
        self._transition(record, RequestStatus.FAILED_RETRYABLE,
                         note=note or f"retryable failure: {classification}",
                         action="RETRY_SAME_LOGICAL_REQUEST")

    def mark_blocking(self, record: LogicalRequestRecord, *, classification: str,
                      reason: str) -> None:
        record.last_provider_classification = classification
        self.blocked_reason = reason
        self._transition(record, RequestStatus.FAILED_BLOCKING, note=reason,
                         action="STOP_AND_RESUME_LATER")

    # --- integrity ----------------------------------------------------------

    def verify_completed(self, archive_reader: Callable[[LogicalRequestRecord], str | None]
                         ) -> list[dict[str, Any]]:
        """Re-hash every completed request's archive against its record.

        Returns one row per completed request. The caller decides what a
        mismatch means; `assert_completed_intact` makes it fatal, which is the
        behaviour L.11 requires of a resume.
        """
        rows: list[dict[str, Any]] = []
        for record in self.completed:
            actual = archive_reader(record)
            rows.append({
                "logical_request_id": record.logical_request_id,
                "recorded_sha256": record.raw_response_sha256,
                "actual_sha256": actual,
                "matches": actual is not None and actual == record.raw_response_sha256,
                "archive_present": actual is not None,
            })
        return rows

    def assert_completed_intact(self,
                                archive_reader: Callable[[LogicalRequestRecord], str | None]
                                ) -> list[dict[str, Any]]:
        rows = self.verify_completed(archive_reader)
        broken = [row for row in rows if not row["matches"]]
        if broken:
            raise CompletedRequestDrift(
                "a completed logical request no longer matches its archive; failing closed "
                "rather than regenerating it, because regeneration would replace the evidence "
                f"that something is wrong: {json.dumps(broken, indent=2)}")
        return rows


def classify_result(result: ProviderGenerationResult | None,
                    validation_all_accepted: bool) -> tuple[RequestStatus, str, str]:
    """Map one `generate_slot` outcome onto a persistent status.

    `generate_slot` has already spent the frozen retry budget internally, so an
    error arriving here means that budget is exhausted for this invocation — not
    that the logical request is dead. A retryable class stays retryable across
    runs, which is what lets a later invocation pick up the same request.
    """
    if result is None:
        return (RequestStatus.FAILED_RETRYABLE, "no_result",
                "the provider returned no result at all")
    if result.error is not None:
        error_class = result.error.error_class
        if error_class in _BLOCKING_STOP_ERRORS:
            return (RequestStatus.FAILED_BLOCKING, error_class.value,
                    "provider reported daily quota exhaustion; stopping cleanly with every "
                    "completed request preserved")
        if error_class in _RETRYABLE_ERRORS:
            return (RequestStatus.FAILED_RETRYABLE, error_class.value,
                    f"retryable provider failure ({error_class.value})")
        return (RequestStatus.FAILED_BLOCKING, error_class.value,
                f"non-retryable provider failure ({error_class.value})")
    if not validation_all_accepted:
        return (RequestStatus.FAILED_RETRYABLE, "invalid_candidate",
                "the response did not validate as exactly 32 accepted candidates")
    return (RequestStatus.COMPLETED_VALID, "OK", "")


__all__ = ["LIVE_STATE_SCHEMA_VERSION", "RequestStatus", "RESUMABLE", "LiveStateError",
           "CompletedRequestDrift", "LogicalRequestRecord", "LiveGenerationState",
           "build_plan", "classify_result"]
