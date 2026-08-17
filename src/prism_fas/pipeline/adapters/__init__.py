"""Stage adapters: the thin layer between the orchestrator and the science.

An adapter's job is to make an existing milestone *executable through the
orchestrator*. It is not allowed to be an implementation. Every adapter here
delegates to the canonical module that already owns the behaviour — the recipe
planner, the route context, the arm schedules, the selector, the locks — and
adds only sequencing, state and provenance around it.

That restriction is the point rather than a style preference. A second
implementation of the selector or the retry policy would be a second scientific
authority, and the two would disagree the moment one was edited. So an adapter
that cannot find its canonical implementation must BLOCK and say so; it must
never supply a replacement.

Three concepts:

* **Mode** — what an adapter was asked to do. A historical milestone verifies;
  C3 additionally has live generation modes.
* **Provider binding** — what a mode is allowed to talk to. `LIVE` reaches the
  real provider and is gated hard; `MOCK` and `REPLAY` are offline fixtures;
  `NONE` cannot make a request at all.
* **Capability** — what a profile permits. The profile decides the ceiling, the
  adapter decides what it needs, and the intersection is what runs.

Separating mode from binding is what lets the smoke profile exercise the whole
live control path — the state machine, checkpointing, retry, resume — while
remaining structurally incapable of reaching Gemini.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from prism_fas.pipeline.profiles import ExecutionProfile
from prism_fas.pipeline.status import RUN_OUTCOME, DualStatus, StatusError


class ProviderBinding(str, Enum):
    """What an adapter run may talk to."""

    NONE = "none"        # cannot make a provider request at all
    MOCK = "mock"        # scripted offline double
    REPLAY = "replay"    # archived responses, replayed offline
    LIVE = "live"        # the real provider; gated by profile AND authorization


#: Bindings that cannot reach a network. Used as a positive allowlist rather
#: than a "not LIVE" check, so a future binding is offline only if it says so.
OFFLINE_BINDINGS: frozenset[ProviderBinding] = frozenset({
    ProviderBinding.NONE, ProviderBinding.MOCK, ProviderBinding.REPLAY})


class AdapterError(RuntimeError):
    """The adapter cannot run as asked."""


class LiveProviderRefused(AdapterError):
    """A live provider binding was requested where it is not permitted.

    Its own class because this is the refusal that protects scientific quota and
    the frozen contract, and callers should be able to catch exactly it.
    """


@dataclass(frozen=True)
class AdapterResult:
    """What one adapter run produced.

    `status` uses the L.8 outcome vocabulary so the row can go straight into the
    master index. The dual status is carried separately because L.3's two axes
    answer a different question from "did this run succeed".
    """

    stage_id: str
    substage: str
    mode: str
    provider_binding: ProviderBinding
    status: str
    status_axes: DualStatus
    summary: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    parent_identities: dict[str, str] = field(default_factory=dict)
    provider_calls: int = 0
    notes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in RUN_OUTCOME:
            raise StatusError(f"adapter status {self.status!r} is not one of {RUN_OUTCOME}")
        if self.provider_binding is ProviderBinding.LIVE and self.provider_calls == 0:
            # Not an error — a live-bound run may legitimately make no call if it
            # resumed a complete archive. Recorded so the combination is visible.
            object.__setattr__(self, "notes", [
                *self.notes,
                "live provider binding made zero calls; every unit was already complete"])

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    @property
    def failed_checks(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if not check.get("ok", False)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "substage": self.substage,
            "mode": self.mode,
            "provider_binding": self.provider_binding.value,
            "status": self.status,
            **self.status_axes.as_dict(),
            "summary": self.summary,
            "checks": list(self.checks),
            "checks_run": len(self.checks),
            "checks_failed": len(self.failed_checks),
            "artifacts": list(self.artifacts),
            "parent_identities": dict(self.parent_identities),
            "provider_calls": self.provider_calls,
            "notes": list(self.notes),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class AdapterRequest:
    """Everything an adapter needs to decide what it may do.

    `authorized_live_generation` is deliberately not derivable from the profile.
    L.2 lets the full profile run scientific work; it does not by itself
    authorize spending live provider quota on the frozen C3 schedule. That
    remains a human decision and arrives here as an explicit flag.
    """

    repo: Path
    profile: ExecutionProfile
    mode: str | None = None
    provider_binding: ProviderBinding | None = None
    resume: bool = False
    authorized_live_generation: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> Any:
        """The execution policy this request runs under, derived from the profile.

        A property rather than a field so it cannot be set to something the
        profile does not permit: a rehearsal request cannot be handed a
        scientific context by a caller that constructs the dataclass directly.
        Derived on access and cheap; `ExecutionContext` is frozen.
        """
        from prism_fas.pipeline.execution import ExecutionContext

        return ExecutionContext.for_profile(self.profile)


class StageAdapter(Protocol):
    """The contract every adapter satisfies."""

    stage_id: str
    substages: tuple[str, ...]

    def default_mode(self, profile: ExecutionProfile) -> str: ...

    def default_binding(self, profile: ExecutionProfile) -> ProviderBinding: ...

    def run(self, request: AdapterRequest) -> list[AdapterResult]: ...


def permitted_bindings(profile: ExecutionProfile, *,
                       authorized_live_generation: bool = False,
                       stage_id: str = "") -> frozenset[ProviderBinding]:
    """Which provider bindings a profile allows, and why.

    Three independent conditions must all hold before LIVE is permitted, and
    each of them is a separate sentence in the contract:

    1. the profile's compute policy must permit a live provider at all (L.2);
    2. the stage must be one the profile names as provider-capable — only C3 is;
    3. a human must have authorized live generation for this run.

    Any one of them missing leaves the offline set. That means a full-profile
    run does NOT acquire live capability merely because the adapter now exists.
    """
    policy = profile.compute_policy
    if policy.live_provider is False:
        return OFFLINE_BINDINGS
    if stage_id and stage_id not in policy.live_provider_permitted_for_stages:
        return OFFLINE_BINDINGS
    if not authorized_live_generation:
        return OFFLINE_BINDINGS
    return OFFLINE_BINDINGS | {ProviderBinding.LIVE}


def assert_binding_permitted(binding: ProviderBinding, request: AdapterRequest, *,
                             stage_id: str) -> None:
    """Fail closed before anything is constructed that could make a request."""
    allowed = permitted_bindings(
        request.profile, authorized_live_generation=request.authorized_live_generation,
        stage_id=stage_id)
    if binding in allowed:
        return
    reasons = []
    policy = request.profile.compute_policy
    if policy.live_provider is False:
        reasons.append(f"profile {request.profile.name!r} forbids a live provider entirely")
    elif stage_id not in policy.live_provider_permitted_for_stages:
        reasons.append(
            f"stage {stage_id} is not in the profile's live_provider_permitted_for_stages "
            f"{list(policy.live_provider_permitted_for_stages)}")
    if not request.authorized_live_generation:
        reasons.append("live generation has not been explicitly authorized for this run")
    raise LiveProviderRefused(
        f"provider binding {binding.value!r} is not permitted for {stage_id}: "
        + "; ".join(reasons))


#: Populated by `prism_fas.pipeline.adapters.registry`. Kept here so importing
#: the package gives access to the protocol without importing every adapter.
__all__ = ["ProviderBinding", "OFFLINE_BINDINGS", "AdapterError", "LiveProviderRefused",
           "AdapterResult", "AdapterRequest", "StageAdapter", "permitted_bindings",
           "assert_binding_permitted"]
