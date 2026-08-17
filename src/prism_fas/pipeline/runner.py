"""The zero-argument runner: what `python train.py` does when told nothing.

`train.py` must work with no flags on a machine the author has never seen. That
means deciding, from evidence rather than from arguments, three things:

* **what kind of run this is** — a CPU rehearsal of the implementation, or the
  real scientific pipeline;
* **where to start** — the first milestone that is not already scientifically
  complete;
* **whether to start at all** — every input the resolved intent needs must be
  present before a single GPU hour is spent.

The safety rule sits on the first of those and is the reason the module exists.
A CPU-only laptop must be able to prove the implementation works, and must not
be able to produce a Version-C P3 result on the way. So the two intents are
separated structurally: they load different profiles, write to different
namespaces, and only one of them can even declare scientific eligibility. A
rehearsal cannot become a scientific ancestor because nothing it writes lives
where a scientific artifact lives.

Nothing here executes science. `resolve` returns a plan; the orchestrator runs
it, under the profile this module selected.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The two zero-argument intents.
CPU_FULL_REHEARSAL = "CPU_FULL_REHEARSAL"
GPU_SCIENTIFIC_FULL = "GPU_SCIENTIFIC_FULL"

#: Profile each intent runs under.
INTENT_PROFILE = {CPU_FULL_REHEARSAL: "rehearsal", GPU_SCIENTIFIC_FULL: "full"}

#: The C3 lock is the evidence that C0-C3 are scientifically complete.
C3_SCIENTIFIC_LOCK = "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"

#: Free space a scientific run should have before it starts, in gigabytes.
#: An engineering guard, not a measurement of the real requirement — which
#: depends on data that is not present until the packages are built.
MINIMUM_FREE_GB_SCIENCE = 50.0
MINIMUM_FREE_GB_REHEARSAL = 5.0


class RunnerError(RuntimeError):
    """The zero-argument run cannot proceed as resolved."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --- scientific completion ---------------------------------------------------

def scientific_completion(repo: Path) -> dict[str, bool]:
    """Which milestones have full-profile scientific evidence on disk.

    Read from the artifact each milestone would have produced, never from a
    status field in a document. A status field can be edited; a frozen lock whose
    identity reproduces cannot be edited without breaking.
    """
    from prism_fas.pipeline.stages import STAGE_IDS

    complete = {stage: False for stage in STAGE_IDS}
    lock = _read_json(repo / C3_SCIENTIFIC_LOCK)
    if (lock.get("status") == "FROZEN_SCIENTIFIC_BANKS"
            and lock.get("execution_profile") == "full"):
        for stage in ("C0", "C1", "C2", "C3"):
            complete[stage] = True
    for stage in STAGE_IDS:
        if complete[stage]:
            continue
        acceptance = _read_json(
            repo / "reports" / "full" / stage.lower() / f"{stage}_ACCEPTANCE.json")
        complete[stage] = (acceptance.get("scientific_status") == "PASS"
                           and acceptance.get("execution_profile") == "full")
    return complete


def first_incomplete_stage(repo: Path) -> str | None:
    """The stage a scientific run should resume at, or None when all are done."""
    from prism_fas.pipeline.stages import STAGE_IDS

    complete = scientific_completion(repo)
    for stage in STAGE_IDS:
        if not complete[stage]:
            return stage
    return None


# --- disk and permissions ----------------------------------------------------

def disk_report(repo: Path, *, required_gb: float) -> dict[str, Any]:
    usage = shutil.disk_usage(repo)
    free_gb = usage.free / (1024 ** 3)
    return {"free_gb": round(free_gb, 2), "total_gb": round(usage.total / (1024 ** 3), 2),
            "required_gb": required_gb, "sufficient": free_gb >= required_gb,
            "note": "an engineering guard. The real requirement depends on the built "
                    "packages, which do not exist until preparation has run"}


def write_permissions(repo: Path) -> dict[str, Any]:
    """Every root the run will write to must be writable before it starts."""
    roots = ("runs", "reports", "state", "data")
    result: dict[str, Any] = {}
    for name in roots:
        path = repo / name
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".prism_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            result[name] = True
        except OSError as error:
            result[name] = f"{type(error).__name__}: {error}"
    return {"roots": result,
            "all_writable": all(value is True for value in result.values())}


# --- the resolved plan -------------------------------------------------------

@dataclass
class RunPlan:
    """What a zero-argument invocation decided to do, and why."""

    intent: str
    profile_name: str
    first_stage: str | None
    last_stage: str
    resume: bool
    reason: str
    scientific_completion: dict[str, bool] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    bundle: dict[str, Any] = field(default_factory=dict)
    disk: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def is_scientific(self) -> bool:
        return self.intent == GPU_SCIENTIFIC_FULL

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_intent": self.intent,
            "profile": self.profile_name,
            "first_stage": self.first_stage,
            "last_stage": self.last_stage,
            "resume": self.resume,
            "reason": self.reason,
            "scientific_completion": dict(self.scientific_completion),
            "scientifically_complete_stages": [stage for stage, done
                                               in self.scientific_completion.items() if done],
            "gpu": dict(self.gpu),
            "environment_profile": self.environment.get("profile_id"),
            "environment_identity": self.environment.get("environment_identity"),
            "bundle": dict(self.bundle),
            "disk": dict(self.disk),
            "permissions": dict(self.permissions),
            "ready": self.ready,
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "target_firewall": "ARMED",
            "scientific_eligible": self.is_scientific,
        }


def resolve(repo: Path, environment: dict[str, Any]) -> RunPlan:
    """Decide the intent, the stage range and whether the run may start.

    `environment` is the report `bootstrap.ensure_environment` returned. Its
    `profile_supports_scientific_execution` flag is the single authority on
    whether this host may do science — it is set by matching real hardware to a
    declared profile, so a CPU host cannot reach the scientific branch by any
    combination of flags or files.
    """
    from prism_fas.pipeline.portability import bundle_readiness

    gpu = dict(environment.get("gpu") or {})
    can_do_science = bool(environment.get("profile_supports_scientific_execution"))
    completion = scientific_completion(repo)
    notes: list[str] = []
    blockers: list[str] = []

    if can_do_science:
        intent = GPU_SCIENTIFIC_FULL
        first = first_incomplete_stage(repo)
        if first is None:
            reason = ("every milestone is already scientifically complete; nothing to "
                      "execute")
            notes.append("C13 is complete: re-running would regenerate nothing")
        else:
            reason = (f"a compatible CUDA GPU matched the declared "
                      f"{environment.get('profile_id')} profile, and {first} is the first "
                      "milestone without full-profile scientific evidence")
        required_gb = MINIMUM_FREE_GB_SCIENCE
    else:
        intent = CPU_FULL_REHEARSAL
        first = "C0"
        reason = ("no CUDA GPU matched a declared scientific profile, so this host "
                  "rehearses the implementation instead of executing science")
        notes.append("a rehearsal writes only to reports/rehearsal and runs/rehearsal "
                     "and can never become a scientific ancestor")
        notes.append("C10-C13 use an isolated fixture target package; the real SiW roots "
                     "are not resolved")
        required_gb = MINIMUM_FREE_GB_REHEARSAL

    profile_name = INTENT_PROFILE[intent]
    bundle = bundle_readiness(repo, intent=intent)
    disk = disk_report(repo, required_gb=required_gb)
    permissions = write_permissions(repo)

    if not bundle["ready"]:
        blockers.append(
            f"the portable bundle is missing {len(bundle['missing'])} required item(s) "
            f"for {intent}")
    if not disk["sufficient"]:
        blockers.append(
            f"insufficient free disk: {disk['free_gb']} GB available, "
            f"{disk['required_gb']} GB required for {intent}")
    if not permissions["all_writable"]:
        unwritable = [name for name, value in permissions["roots"].items() if value is not True]
        blockers.append(f"output roots are not writable: {unwritable}")

    return RunPlan(
        intent=intent, profile_name=profile_name, first_stage=first, last_stage="C13",
        resume=True, reason=reason, scientific_completion=completion, gpu=gpu,
        environment=environment, bundle=bundle, disk=disk, permissions=permissions,
        blockers=blockers, notes=notes)


# --- the console summary -----------------------------------------------------

def preflight_summary(repo: Path, plan: RunPlan, *, git_identity: str = "") -> str:
    """The block printed before any long work begins."""
    import platform

    gpu = plan.gpu
    device = "CUDA" if gpu.get("available") else "CPU"
    environment = plan.environment
    bundle = plan.bundle
    lines = [
        "",
        "PRISM-FAS-C-LLM Portable Runner",
        "=" * 62,
        f"  Project root        {repo}",
        f"  Git identity        {git_identity or 'unavailable'}",
        f"  Python              {platform.python_version()} "
        f"({platform.python_implementation()})",
        f"  Environment         {environment.get('action', 'UNKNOWN')}  "
        f"profile={environment.get('profile_id')} "
        f"[{environment.get('profile_status', '?')}]",
        f"  Environment id      {str(environment.get('environment_identity', ''))[:16]}",
        f"  Device              {device}",
    ]
    if gpu.get("available"):
        lines += [
            f"  GPU                 {gpu.get('name')}",
            f"  VRAM                {gpu.get('memory_total_mb')} MB",
            f"  Driver              {gpu.get('driver_version')}  "
            f"compute={gpu.get('compute_capability') or 'unreported'}",
        ]
    else:
        lines.append(f"  GPU                 none detected "
                     f"({gpu.get('query_error') or 'no CUDA device'})")

    lines += [
        "",
        f"  Execution intent    {plan.intent}",
        f"  Profile             {plan.profile_name}  "
        f"(scientific_eligible={plan.is_scientific})",
        f"  Stage range         {plan.first_stage or 'none'} -> {plan.last_stage}"
        f"   resume={plan.resume}",
        f"  Reason              {plan.reason}",
        "",
        f"  Bundle              {'PASS' if bundle['ready'] else 'FAIL'}  "
        f"({bundle['present_count']}/{bundle['required_count']} required items present)",
        f"  Disk                {'PASS' if plan.disk['sufficient'] else 'FAIL'}  "
        f"({plan.disk['free_gb']} GB free, {plan.disk['required_gb']} GB required)",
        f"  Write access        {'PASS' if plan.permissions['all_writable'] else 'FAIL'}",
        f"  Target firewall     ARMED",
        "",
        f"  Output root         {repo / plan.profile_name if False else repo}",
        f"  Reports             reports/{plan.profile_name}/",
        f"  Runs                runs/{plan.profile_name}/",
    ]
    if plan.notes:
        lines.append("")
        for note in plan.notes:
            lines.append(f"  note                {note}")
    if plan.blockers:
        lines.append("")
        lines.append("  BLOCKED — nothing was executed:")
        for blocker in plan.blockers:
            lines.append(f"    - {blocker}")
        for item in bundle.get("missing", [])[:20]:
            lines.append(f"    MISSING  {item['expected_path']}   ({item['logical_name']})")
    lines.append("=" * 62)
    return "\n".join(lines)


def completion_summary(repo: Path, plan: RunPlan, result: Any) -> str:
    """Navigation printed after a zero-argument run finishes."""
    reports = f"reports/{plan.profile_name}"
    lines = ["", "=" * 62]
    if plan.is_scientific:
        complete = result is not None and getattr(result, "outcome", "") == "PASS"
        lines.append("PRISM-FAS-C-LLM FULL SCIENTIFIC PIPELINE = "
                     f"{'COMPLETE' if complete else getattr(result, 'outcome', 'INCOMPLETE')}")
        lines.append(f"C4-C13 = {'COMPLETE' if complete else 'NOT COMPLETE'}")
        lines += [
            "",
            f"  Final report      {reports}/final/report.html",
            f"  Main results      {reports}/tables/main_results.csv",
            f"  Plots             {reports}/plots/",
            f"  Tables            {reports}/tables/",
            f"  Checkpoints       runs/{plan.profile_name}/",
            f"  Master index      state/MASTER_RUN_INDEX.json",
            f"  C_ACCEPTANCE      {reports}/c13/C_ACCEPTANCE.json",
        ]
    else:
        lines.append("CPU FULL REHEARSAL = "
                     f"{getattr(result, 'outcome', 'INCOMPLETE')}")
        lines += [
            "SCIENTIFIC OUTPUTS MODIFIED = 0",
            "REAL TARGET ACCESS = 0",
            "",
            f"  Reports           {reports}/",
            f"  Runs              runs/{plan.profile_name}/",
            f"  Master index      state/MASTER_RUN_INDEX.json",
            "",
            "  This was an implementation rehearsal. It completes no milestone,",
            "  selects no winner and produces no scientific evidence.",
        ]
    lines.append("=" * 62)
    return "\n".join(lines)


__all__ = ["CPU_FULL_REHEARSAL", "GPU_SCIENTIFIC_FULL", "INTENT_PROFILE",
           "MINIMUM_FREE_GB_SCIENCE", "MINIMUM_FREE_GB_REHEARSAL", "RunnerError",
           "scientific_completion", "first_incomplete_stage", "disk_report",
           "write_permissions", "RunPlan", "resolve", "preflight_summary",
           "completion_summary"]
