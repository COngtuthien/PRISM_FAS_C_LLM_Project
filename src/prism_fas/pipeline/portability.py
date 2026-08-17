"""Backend portability: what may change with the GPU, and what may not.

Version C is executed on constrained compute now and on a different, sufficiently
resourced GPU later. L.12 and the v1.4 compute-portability contract draw the line
this module enforces: *engineering* settings — device, physical microbatch,
gradient accumulation, worker count, I/O tuning — adapt to the machine; the
*scientific* content — effective batch composition and order, model topology,
loss semantics, seeds, identities — does not.

The failure this prevents is quiet and expensive: moving to a smaller card,
halving the batch to make it fit, and shipping a result that is no longer the
preregistered experiment. So the two questions are answered separately.

* `resolve_microbatch` answers "what physically fits" and is free to differ per
  backend. It only ever returns a split whose product reconstitutes the frozen
  effective batch exactly; it will refuse rather than round.
* `scientific_identity_material` answers "what was actually run" and is
  deliberately blind to every operational field. Two runs of the same experiment
  on Modal and on a lab GPU must produce the same value, or one of them has
  quietly changed the science.

The batch composition itself is not declared here. It belongs to
`prism_fas.detector.sampler.BatchContract`, which already owns and validates it,
and this module reads it from there rather than restating it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "prism-backend-portability-v1"

#: Operational provenance. Recorded on every artifact, and excluded from every
#: scientific identity — the spec requires backend information to be provenance,
#: never a treatment factor (L.12).
OPERATIONAL_FIELDS: tuple[str, ...] = (
    # The namespace every backend fact is recorded under. Dropping the whole key
    # is what makes the guarantee total: a field added to a BackendProfile later
    # cannot leak into an identity by being forgotten in this list.
    "operational_provenance", "backend_profile",
    "backend", "backend_name", "device", "device_name", "gpu_model", "gpu_uuid",
    "gpu_vendor", "vram_gb", "supports_amp", "workspace", "modal_workspace",
    "modal_account", "hostname", "host", "filesystem_root", "repo_root",
    "num_workers", "workers", "physical_microbatch", "microbatch",
    "gradient_accumulation_steps", "pin_memory", "prefetch_factor", "io_threads",
    "cuda_version", "driver_version", "wall_clock_seconds", "cost_usd",
    "started_at_utc", "finished_at_utc",
)

#: The single key a backend is stamped under. Named here so writers and the
#: identity stripper cannot disagree about where machine facts live.
PROVENANCE_NAMESPACE = "operational_provenance"

#: Patterns that make a checkpoint unrestorable somewhere else. Each is a real
#: portability defect rather than a style preference: a checkpoint carrying any
#: of them cannot be resolved on the collaborator's machine.
NON_PORTABLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^[A-Za-z]:[\\/]", "an absolute Windows drive path"),
    (r"^/(?:root|home|mnt|modal|vol|workspace|content)/", "an absolute POSIX mount path"),
    (r"GPU-[0-9a-fA-F]{8}-", "a physical GPU UUID"),
    (r"\\\\[^\\]+\\", "a UNC network share"),
)

#: Keys whose values are allowed to look like machine paths because they are
#: explicitly operational provenance rather than a resolution input.
PROVENANCE_KEYS: frozenset[str] = frozenset({
    "recorded_filesystem_root", "recorded_device", "recorded_hostname",
    "operational_provenance", "environment", "compute",
})


class PortabilityError(RuntimeError):
    """A backend adaptation would change scientific content."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BackendProfile:
    """One execution backend, described only in operational terms.

    There is no scientific field on this object, and that absence is the point:
    nothing a `BackendProfile` carries can reach an identity or a metric.
    """

    name: str
    device: str = "cpu"
    vram_gb: float | None = None
    workers: int = 0
    supports_amp: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "device": self.device, "vram_gb": self.vram_gb,
                "workers": self.workers, "supports_amp": self.supports_amp,
                "notes": self.notes, "classification": "OPERATIONAL_PROVENANCE_ONLY"}


#: Backends the repository knows how to describe. Adding one is an engineering
#: change; none of them may alter a frozen contract.
KNOWN_BACKENDS: dict[str, BackendProfile] = {
    "local_cpu": BackendProfile("local_cpu", device="cpu", vram_gb=None, workers=0,
                                notes="engineering fixtures and smoke only"),
    "modal": BackendProfile("modal", device="cuda", vram_gb=24.0, workers=4,
                            supports_amp=True, notes="metered; preserved for later use"),
    "ssh_lab": BackendProfile("ssh_lab", device="cuda", vram_gb=24.0, workers=4,
                              supports_amp=True, notes="collaborator GPU"),
    "collaborator_gpu": BackendProfile("collaborator_gpu", device="cuda", vram_gb=None,
                                       workers=4, supports_amp=True,
                                       notes="VRAM resolved at run time on the host"),
}


@dataclass(frozen=True)
class MicrobatchPlan:
    """A physical split of one frozen effective batch.

    `effective_batch` and `composition` are inputs, never outputs: this object
    reports how the frozen batch was divided, and it cannot report a different
    batch than the one it was given.
    """

    effective_batch: int
    composition: dict[str, int]
    microbatch: int
    accumulation_steps: int
    backend: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.microbatch * self.accumulation_steps != self.effective_batch:
            raise PortabilityError(
                f"microbatch {self.microbatch} x accumulation {self.accumulation_steps} "
                f"= {self.microbatch * self.accumulation_steps}, which is not the frozen "
                f"effective batch {self.effective_batch}. A backend may change how a batch "
                "is divided; it may never change the batch")

    @property
    def preserves_effective_batch(self) -> bool:
        return self.microbatch * self.accumulation_steps == self.effective_batch

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_batch": self.effective_batch,
            "effective_composition": dict(self.composition),
            "physical_microbatch": self.microbatch,
            "gradient_accumulation_steps": self.accumulation_steps,
            "backend": self.backend,
            "preserves_effective_batch": self.preserves_effective_batch,
            "classification": "ENGINEERING_ADAPTIVE",
            "contract": ("§15.2.2: microbatch may change only with gradient accumulation "
                         "preserving effective batch and order; effective scientific batch "
                         "composition remains 12 real live + 12 real spoof + 8 synthetic"),
            "reason": self.reason,
        }


def frozen_composition() -> dict[str, int]:
    """The effective batch composition, read from its canonical owner."""
    from prism_fas.detector.sampler import DEFAULT_COMPOSITION

    return dict(DEFAULT_COMPOSITION)


def divisors(value: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(1, value + 1) if value % candidate == 0)


def resolve_microbatch(*, backend: BackendProfile, effective_batch: int | None = None,
                       composition: Mapping[str, int] | None = None,
                       max_microbatch: int | None = None) -> MicrobatchPlan:
    """The largest physical microbatch that fits and still divides the batch.

    Two constraints, both hard. The microbatch must divide the effective batch
    exactly, so accumulation reconstitutes it without a remainder step that would
    see a differently composed final micro-batch. And it must not exceed what the
    backend can hold. When those two cannot both be satisfied the function raises
    rather than shrinking the batch — L.12 forbids silently shrinking a scientific
    budget because the hardware is small.
    """
    parts = dict(composition or frozen_composition())
    total = int(effective_batch if effective_batch is not None else sum(parts.values()))
    if total <= 0:
        raise PortabilityError("the effective batch must be positive")

    ceiling = total
    if max_microbatch is not None:
        ceiling = min(ceiling, int(max_microbatch))
    if backend.vram_gb is not None:
        # A deliberately coarse heuristic. It is engineering guidance, not a
        # measurement, and it can only ever make the microbatch smaller — never
        # the batch.
        ceiling = min(ceiling, max(1, int(backend.vram_gb // 1.5)))

    candidates = [item for item in divisors(total) if item <= ceiling]
    if not candidates:
        raise PortabilityError(
            f"no microbatch divides the frozen effective batch {total} within the "
            f"{ceiling} the {backend.name!r} backend allows. The batch is a frozen "
            "scientific quantity and is not reduced to fit (L.12); use a backend that "
            "can hold at least one divisor")
    microbatch = max(candidates)
    return MicrobatchPlan(
        effective_batch=total, composition=parts, microbatch=microbatch,
        accumulation_steps=total // microbatch, backend=backend.name,
        reason=f"largest divisor of {total} not exceeding the backend ceiling {ceiling}")


def assert_composition_preserved(plan: MicrobatchPlan,
                                 expected: Mapping[str, int] | None = None) -> None:
    """Refuse a plan whose composition drifted from the frozen contract."""
    frozen = dict(expected or frozen_composition())
    if dict(plan.composition) != frozen:
        raise PortabilityError(
            f"batch composition {plan.composition} is not the frozen composition {frozen}; "
            "a backend change may not alter what a batch contains")
    if sum(frozen.values()) != plan.effective_batch:
        raise PortabilityError(
            f"effective batch {plan.effective_batch} does not equal the frozen composition "
            f"total {sum(frozen.values())}")


def scientific_identity_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The payload with every operational field removed, recursively.

    This is what a scientific identity is computed over. If adding a backend to
    the record changes the identity, the backend has become part of the science,
    which is exactly what L.12 forbids.
    """
    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: strip(item) for key, item in value.items()
                    if key not in OPERATIONAL_FIELDS}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    return strip(dict(payload))


def scientific_identity(payload: Mapping[str, Any]) -> str:
    return _sha(_canonical(scientific_identity_material(payload)))


def identity_is_backend_invariant(payload: Mapping[str, Any],
                                  backends: Iterable[BackendProfile]) -> dict[str, Any]:
    """Prove the identity does not move when only the backend moves.

    Rather than assert the property, this computes it: the same scientific
    payload is stamped with each backend in turn and the identities compared. A
    single differing value names the backend that leaked into the science.

    The backend is stamped exactly the way the pipeline records it — nested under
    `PROVENANCE_NAMESPACE` — because that is the arrangement the guarantee has to
    hold for. Merging machine facts into the payload's own top level would not be
    a stricter test; it would be testing a layout the writers never produce.
    """
    identities: dict[str, str] = {}
    for backend in backends:
        stamped = {**dict(payload), PROVENANCE_NAMESPACE: backend.as_dict()}
        identities[backend.name] = scientific_identity(stamped)
    unique = sorted(set(identities.values()))
    return {
        "identities": identities,
        "invariant": len(unique) == 1,
        "distinct_identity_count": len(unique),
        "baseline_identity": scientific_identity(payload),
        "meaning": ("a scientific identity must not depend on the backend it was computed "
                    "on; more than one distinct value means backend information reached "
                    "the scientific payload (L.12)"),
    }


def _non_portable_findings(value: str) -> list[str]:
    return [reason for pattern, reason in NON_PORTABLE_PATTERNS
            if re.search(pattern, value)]


def checkpoint_portability_audit(payload: Mapping[str, Any], *,
                                 path: str = "") -> dict[str, Any]:
    """Find every unrecoverable machine dependency in a checkpoint record.

    A checkpoint must restore on a machine that has never seen this one, so any
    absolute path, mount point, GPU UUID or host name it needs in order to
    resolve its inputs is a defect. Values under an explicitly operational key
    are exempt: recording *where* a run happened is provenance, and only using
    that value as a resolution input is the problem.
    """
    findings: list[dict[str, Any]] = []

    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                walk(item, (*trail, str(key)))
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, (*trail, f"[{index}]"))
        elif isinstance(node, str):
            if any(part in PROVENANCE_KEYS for part in trail):
                return
            for reason in _non_portable_findings(node):
                findings.append({"key_path": ".".join(trail), "value": node,
                                 "reason": reason})

    walk(dict(payload), ())
    return {
        "schema_version": SCHEMA_VERSION,
        "path": path,
        "portable": not findings,
        "findings": findings,
        "checked_patterns": [reason for _pattern, reason in NON_PORTABLE_PATTERNS],
        "exempt_provenance_keys": sorted(PROVENANCE_KEYS),
        "meaning": ("a checkpoint must resolve its inputs through portable logical names "
                    "and config identities. Recording the machine it ran on is provenance; "
                    "needing that machine to restore is a defect"),
    }


@dataclass(frozen=True)
class LogicalPath:
    """A path expressed as root name plus relative path, not as a location."""

    root: str
    relative: str

    def as_dict(self) -> dict[str, Any]:
        return {"root": self.root, "relative": self.relative,
                "logical": f"{{{self.root}}}/{self.relative}"}


def portable_path(path: str, roots: Mapping[str, str]) -> LogicalPath:
    """Rewrite an absolute path as a logical one against declared roots.

    The longest matching root wins, so a nested root does not lose to its parent.
    A path under no declared root is an error rather than a pass-through: silently
    keeping it absolute is how an unportable checkpoint gets written.
    """
    normalized = str(path).replace("\\", "/")
    best: tuple[str, str] | None = None
    for name, value in roots.items():
        candidate = str(value).replace("\\", "/").rstrip("/")
        if candidate and normalized.lower().startswith(candidate.lower() + "/"):
            if best is None or len(candidate) > len(best[1]):
                best = (name, candidate)
    if best is None:
        raise PortabilityError(
            f"{path!r} lies under none of the declared roots {sorted(roots)}; it cannot be "
            "expressed portably and would not resolve on another backend")
    return LogicalPath(root=best[0], relative=normalized[len(best[1]) + 1:])


def backend_report(backend: BackendProfile, plan: MicrobatchPlan) -> dict[str, Any]:
    """One record binding an operational backend to the frozen batch it ran.

    The backend goes under the provenance namespace rather than at the top
    level, so this record can be embedded in an artifact without its machine
    facts reaching that artifact's scientific identity.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        PROVENANCE_NAMESPACE: backend.as_dict(),
        "microbatch_plan": plan.as_dict(),
        "scientific_fields_affected": [],
        "contract": ("backend switching is governed by the compute portability/precision "
                     "identity contract; cost pressure does not authorize mixing "
                     "incompatible scientific compute profiles (L.12)"),
    }


__all__ = ["SCHEMA_VERSION", "OPERATIONAL_FIELDS", "PROVENANCE_NAMESPACE",
           "NON_PORTABLE_PATTERNS",
           "PROVENANCE_KEYS", "PortabilityError", "BackendProfile", "KNOWN_BACKENDS",
           "MicrobatchPlan", "frozen_composition", "divisors", "resolve_microbatch",
           "assert_composition_preserved", "scientific_identity_material",
           "scientific_identity", "identity_is_backend_invariant",
           "checkpoint_portability_audit", "LogicalPath", "portable_path",
           "backend_report"]
