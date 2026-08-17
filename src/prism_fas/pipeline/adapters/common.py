"""Shared machinery for the C4-C13 engineering adapters.

Every adapter from C4 onward has the same shape and the same three refusals, so
they live here once rather than being retyped ten times with ten opportunities to
drift.

The shape: an adapter declares what it needs in order to run *scientifically*,
and what it can rehearse on fixtures. Under `full` it checks the real inputs and
BLOCKS naming each missing one; under `smoke` it executes the same control flow
against tiny fixtures and stamps everything `scientific_eligible=false`.

The three refusals:

* **No provider.** None of C4-C13 talks to an LLM. They bind `ProviderBinding.NONE`
  and never construct a provider, which is why a smoke run over the whole
  pipeline cannot spend a single Gemini call.
* **No fixture may masquerade as science.** A fixture-backed result is marked at
  the artifact level — `fixture_backed`, `substituted_components` — and the full
  profile refuses to accept one. The difference between "we ran the topology on a
  random tower" and "we ran the frozen SigLIP2" is the difference between an
  engineering pass and a scientific claim.
* **No silent shrinking.** Smoke reduces samples, steps, epochs and seeds through
  the profile's declared budget and nothing else. A reduction of anything the
  profile lists as preserved is refused by `prism_fas.pipeline.budget`.

An adapter here owns sequencing, gating and provenance. It owns no science: every
model, loss, gate, metric, lock and selector is imported from the module that
already implements it, and an adapter that cannot import its canonical
implementation BLOCKS rather than supplying a replacement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from prism_fas.pipeline.adapters import (AdapterError, AdapterRequest, AdapterResult,
                                         ProviderBinding, assert_binding_permitted)
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.state import atomic_write_json
from prism_fas.pipeline.status import DualStatus

#: Marker every fixture-backed artifact carries. Grep-able on purpose: it must be
#: possible to find every engineering rehearsal in the tree with one search.
FIXTURE_MARKER = "ENGINEERING_FIXTURE_NOT_SCIENTIFIC_EVIDENCE"

#: The reason string a stage uses when the full profile lacks a real input.
MISSING_INPUT = "REQUIRED_SCIENTIFIC_INPUT_ABSENT"


class CanonicalImplementationUnavailable(AdapterError):
    """The canonical module an adapter delegates to could not be imported.

    Its own type because the correct response is to block, never to substitute.
    An adapter that reimplemented the thing it could not import would become a
    second scientific authority, and the two would disagree the moment one moved.
    """


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check(check_id: str, ok: bool, summary: str, **detail: Any) -> dict[str, Any]:
    return {"check_id": check_id, "ok": bool(ok), "summary": summary, "detail": detail}


def stage_reports_dir(request: AdapterRequest, stage_id: str) -> Path:
    """Where this stage's evidence goes, under the running profile's namespace.

    The namespace comes from the profile rather than from the adapter, which is
    what keeps a smoke rehearsal out of `reports/full/` structurally rather than
    by convention.
    """
    return request.repo / request.profile.reports_namespace / stage_id.lower()


def stage_runs_dir(request: AdapterRequest, stage_id: str) -> Path | None:
    runs = request.profile.runs_dir(request.repo)
    return None if runs is None else runs / stage_id.lower()


def write_artifact(request: AdapterRequest, path: Path,
                   payload: dict[str, Any]) -> str:
    """Write one stage artifact, stamped with its profile and eligibility.

    Goes through `profile.stamp()` so no adapter can write an artifact that
    omits `execution_profile` or `scientific_eligible`; `assert_not_promoted`
    inside the state writer refuses a non-full artifact claiming eligibility.
    """
    body = {**request.profile.stamp(), **payload}
    if not request.profile.scientific_eligible:
        body.setdefault("artifact_kind", FIXTURE_MARKER)
    atomic_write_json(path, body)
    return path.relative_to(request.repo).as_posix()


@dataclass(frozen=True)
class RequiredInput:
    """One artifact the full profile needs before this stage can run for real."""

    name: str
    relative_path: str
    description: str
    #: True when the absence is a hard block rather than a warning.
    blocking: bool = True

    def resolve(self, repo: Path) -> dict[str, Any]:
        path = repo / self.relative_path
        present = path.exists()
        return {"name": self.name, "path": self.relative_path, "present": present,
                "blocking": self.blocking and not present,
                "description": self.description}


@dataclass
class SmokeBudget:
    """The profile's engineering budget, read rather than chosen.

    L.12 permits smoke to reduce samples, steps, epochs and seeds *only* through
    the profile's declared budget. Every number an adapter uses for a tiny run
    comes from here, so the reduction is auditable from the profile config and
    cannot be hidden in an adapter.
    """

    samples: int = 8
    steps: int = 2
    epochs: int = 1
    seeds: int = 1
    wall_clock_seconds: int = 1800

    @classmethod
    def from_profile(cls, profile: Any) -> "SmokeBudget":
        budget = dict(profile.engineering_budget or {})
        return cls(
            samples=int(budget.get("max_samples_per_split", 8)),
            steps=int(budget.get("max_steps_per_epoch", 2)),
            epochs=int(budget.get("max_epochs", 1)),
            seeds=int(budget.get("max_seeds", 1)),
            wall_clock_seconds=int(budget.get("max_wall_clock_seconds", 1800)))

    def as_dict(self) -> dict[str, Any]:
        return {"max_samples_per_split": self.samples, "max_steps_per_epoch": self.steps,
                "max_epochs": self.epochs, "max_seeds": self.seeds,
                "max_wall_clock_seconds": self.wall_clock_seconds,
                "source": "the profile's declared engineering_budget (L.12)",
                "reducible_dimensions": ["samples", "steps", "epochs", "seeds"],
                "preserved": ["code_paths", "tensor_semantics", "model_topology",
                              "active_loss_semantics", "target_firewall",
                              "artifact_schemas"]}


@dataclass
class EngineeringAdapter:
    """Base for the C4-C13 stage adapters.

    Subclasses supply `required_inputs` and one method per mode. This class owns
    the parts that must be identical everywhere: binding refusal, the full-profile
    precondition gate, artifact stamping and result assembly.
    """

    stage_id: str = ""
    substages: tuple[str, ...] = ()
    title: str = ""
    #: Modes this adapter can run, in the order a full pass would use them.
    modes: tuple[str, ...] = ()
    #: True when scientific execution of this stage needs a real accelerator.
    requires_gpu: bool = False

    # --- protocol -------------------------------------------------------------

    def default_mode(self, profile: Any) -> str:
        return self.modes[0] if self.modes else "VERIFY"

    def default_binding(self, profile: Any) -> ProviderBinding:
        # Not a function of the profile: no stage from C4 onward has an execution
        # path that could use a provider, so no profile can grant it one.
        return ProviderBinding.NONE

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return ()

    # --- overridden by subclasses --------------------------------------------

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        """The stage's control flow. ONE implementation, both execution paths.

        There is deliberately no `run_full` here to override. A stage used to get
        a default scientific method that refused with SCIENTIFIC_PATH_NOT_EXERCISED,
        which meant the rehearsal exercised code the scientific run would never
        reach — the two could not drift apart, because they were never together.

        Now the context decides what this method does: whether a fixture may
        stand in, how much may be truncated, where artifacts go, whether a
        governing lock may be written. Everything else — scheduling, identity,
        checkpointing, resume, selection, acceptance, the writers — is the same
        code under both, so rehearsing it is evidence about the scientific path
        rather than about a parallel one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement workflow(); every C4-C13 "
            "adapter must, because there is no placeholder to fall back to")

    # --- shared behaviour -----------------------------------------------------

    def run(self, request: AdapterRequest) -> list[AdapterResult]:
        binding = self.default_binding(request.profile)
        assert_binding_permitted(binding, request, stage_id=self.stage_id)

        context = ExecutionContext.for_profile(request.profile)
        if context.is_scientific:
            # The gate is what a scientific run legitimately stops on: a missing
            # dataset, weight, accelerator. It is never "the code is not written".
            gate = self.full_precondition_gate(request)
            if gate is not None:
                return [gate]
        return list(self.workflow(request, context))

    def full_precondition_gate(self, request: AdapterRequest) -> AdapterResult | None:
        """Under full, refuse to start without the real inputs. Name every one.

        Returning BLOCKED here rather than falling back to a fixture is the whole
        contract: a full-profile artifact built on a fixture would be a
        scientifically eligible claim about something that never ran.
        """
        resolved = [item.resolve(request.repo) for item in self.required_inputs()]
        missing = [item for item in resolved if item["blocking"]]

        checks = [check(f"{self.stage_id.lower()}_input_{item['name']}", item["present"],
                        f"{item['path']} is {'present' if item['present'] else 'absent'}",
                        **item) for item in resolved]

        if self.requires_gpu:
            available, detail = _accelerator_available()
            checks.append(check(
                f"{self.stage_id.lower()}_accelerator", available,
                "an accelerator is available for scientific training" if available else
                "no accelerator is available; the full profile declares gpu: required",
                **detail))
            if not available:
                missing.append({"name": "accelerator", "path": "<gpu>", "present": False,
                                "blocking": True,
                                "description": "scientific training requires a real GPU; "
                                               "L.12 forbids shrinking the scientific "
                                               "budget to fit the machine"})

        if not missing:
            return None
        names = ", ".join(item["name"] for item in missing)
        return AdapterResult(
            stage_id=self.stage_id, substage=self.stage_id, mode="FULL_PRECONDITION_GATE",
            provider_binding=ProviderBinding.NONE, status="BLOCKED",
            status_axes=DualStatus(engineering="BLOCKED", scientific="BLOCKED"),
            summary=f"{self.stage_id} cannot execute scientifically here: {names}",
            checks=checks, provider_calls=0,
            notes=[f"{MISSING_INPUT}: {names}",
                   "the stage is engineering-ready; it is BLOCKED on inputs, not on code",
                   "no fixture is substituted under the full profile"],
            detail={"missing_inputs": missing,
                    "resolution": "supply these on the execution backend and rerun "
                                  "`python train.py --profile full --from "
                                  f"{self.stage_id} --to {self.stage_id} --resume`"})

    def blocked(self, request: AdapterRequest, reason_code: str, summary: str,
                checks: Sequence[dict[str, Any]] = (), **detail: Any) -> AdapterResult:
        return AdapterResult(
            stage_id=self.stage_id, substage=self.stage_id, mode=reason_code,
            provider_binding=ProviderBinding.NONE, status="BLOCKED",
            status_axes=DualStatus(engineering="BLOCKED", scientific="BLOCKED"),
            summary=summary, checks=list(checks), provider_calls=0,
            notes=[reason_code], detail=dict(detail))

    def result(self, request: AdapterRequest, *, mode: str,
               checks: Sequence[dict[str, Any]], artifacts: Sequence[str] = (),
               summary: str = "", notes: Sequence[str] = (),
               parent_identities: dict[str, str] | None = None,
               substage: str | None = None, **detail: Any) -> AdapterResult:
        """Assemble one adapter result from its checks.

        The dual status is derived, not passed in: a smoke run that passed is
        SMOKE_PASS on the engineering axis and NOT_RUN on the scientific one, and
        no argument to this method can produce any other combination. That is
        what makes "smoke is not scientific completion" structural.
        """
        failed = [item for item in checks if not item["ok"]]
        # L.3 fixes the engineering vocabulary at NOT_TESTED | RUNNING | SMOKE_PASS
        # | SMOKE_FAIL | BLOCKED. Both `smoke` and `rehearsal` are non-eligible
        # profiles that EXECUTE the code path, which is exactly what SMOKE_PASS
        # describes. Inventing a sixth value for the rehearsal would edit a frozen
        # vocabulary; leaving it NOT_TESTED would claim the path was never run.
        smoke = request.profile.name in ("smoke", "rehearsal")
        status = "PASS" if not failed else ("FAIL" if smoke else "BLOCKED")
        engineering = ("SMOKE_PASS" if smoke and not failed else
                       "SMOKE_FAIL" if smoke else
                       "NOT_TESTED" if not failed else "BLOCKED")
        return AdapterResult(
            stage_id=self.stage_id, substage=substage or self.stage_id, mode=mode,
            provider_binding=ProviderBinding.NONE, status=status,
            status_axes=DualStatus(engineering=engineering, scientific="NOT_RUN"),
            summary=summary or f"{self.stage_id} {mode.lower().replace('_', ' ')}"
                               f" — {len(checks) - len(failed)}/{len(checks)} checks passed",
            checks=list(checks), artifacts=list(artifacts),
            parent_identities=dict(parent_identities or {}), provider_calls=0,
            notes=list(notes), detail=dict(detail))


def _accelerator_available() -> tuple[bool, dict[str, Any]]:
    try:
        import torch
    except ImportError as error:  # torch absent is itself the answer
        return False, {"torch": f"unavailable: {error}"}
    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps = bool(getattr(getattr(torch, "backends", None), "mps", None)
               and torch.backends.mps.is_available())
    return (cuda or mps), {"torch_version": torch.__version__, "cuda_available": cuda,
                           "mps_available": mps}


def import_canonical(dotted: str, name: str) -> Any:
    """Import one canonical implementation, or block naming what is missing."""
    import importlib

    try:
        module = importlib.import_module(dotted)
    except ImportError as error:
        raise CanonicalImplementationUnavailable(
            f"{dotted} does not import; it is the canonical implementation of {name!r} "
            f"and this adapter has no substitute for it ({error})") from error
    return module


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def resume_decision(request: AdapterRequest, unit_id: str, artifact: Path, *,
                    expected_identity: str | None,
                    identity_key: str = "identity") -> dict[str, Any]:
    """Should this unit be skipped? Identity-aware, never "the file exists".

    L.11 requires four conditions before a completed unit is reused. Here the
    artifact's own recorded identity must match what the caller expects; a file
    that exists but records a different identity means the inputs moved, which is
    a fail-closed condition rather than a reason to reuse it.
    """
    payload = read_json(artifact)
    present = payload is not None
    recorded = (payload or {}).get(identity_key)
    matches = bool(expected_identity) and recorded == expected_identity
    skip = bool(request.resume and present and matches)
    return {
        "unit_id": unit_id,
        "artifact": artifact.name,
        "present": present,
        "recorded_identity": recorded,
        "expected_identity": expected_identity,
        "identity_matches": matches,
        "resume_requested": request.resume,
        "action": "SKIP_VALID_COMPLETE" if skip else "EXECUTE",
        "reason": ("the recorded identity validates; the completed unit is reused "
                   "unchanged (L.11)" if skip else
                   "no validated completion exists for this identity; executing"),
    }


__all__ = ["FIXTURE_MARKER", "MISSING_INPUT", "CanonicalImplementationUnavailable",
           "utc", "check", "stage_reports_dir", "stage_runs_dir", "write_artifact",
           "RequiredInput", "SmokeBudget", "EngineeringAdapter", "ExecutionContext",
           "import_canonical",
           "read_json", "resume_decision"]
