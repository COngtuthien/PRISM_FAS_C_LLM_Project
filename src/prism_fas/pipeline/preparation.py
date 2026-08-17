"""Rebuilding the derived data trees, automatically, before C4.

`data/processed` and `data/packages` are deterministic derivations of the raw
datasets. They are large, so they are not copied between machines; they are also
required by every scientific stage from C4 onward. Previously that combination
was recorded as "does not travel in the folder", which quietly made the
collaborator responsible for a manual preparation step.

This module removes that step. If a derived tree is absent and its raw source is
present, `python train.py` builds it, validates it, and continues into C4 — no
second command, no flags.

Four properties matter here, and each is delegated rather than reimplemented:

**Deterministic.** Every builder is the one the project already uses
(`prism_fas.data.m2_runner`, `prism_fas.data.package`, `m3b`,
`prism_fas.synthesis.pair_plan`). None of them is re-derived here; this module
decides *whether* to call them and in what order.

**Resumable.** Every builder takes `resume=True` and skips work whose hash still
validates. An interrupted preparation continues; a valid tree is never rebuilt
because the process restarted, and never because the absolute root changed.

**Validated.** A build is followed by the package's own validator and lock. A
tree that exists but does not validate is a failure, not a pass.

**Honest about absence.** With no raw source there is nothing to derive from, and
this module reports MISSING_RAW_DATA rather than fabricating a smaller package.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-preparation-v1"

MISSING_RAW_DATA = "MISSING_RAW_DATA"
PREPARATION_FAILED = "PREPARATION_FAILED"

#: Built in this order; each consumes the previous one's output.
STEPS = ("m2_preprocess", "m3a_package", "m3b_priors", "gpat_pairs")


class PreparationError(RuntimeError):
    def __init__(self, reason: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class StepOutcome:
    name: str
    action: str          # BUILT | REUSED_VALID | SKIPPED_NOT_NEEDED | FAILED
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.name, "action": self.action, "summary": self.summary,
                "seconds": round(self.seconds, 2), **self.detail}


def what_is_needed(repo: Path) -> dict[str, Any]:
    """Which derived trees are absent, and whether they can be rebuilt.

    Reported without building anything, so the preflight can print it before a
    long run rather than discovering it at C4.
    """
    from prism_fas.pipeline import portable_paths

    resolution = portable_paths.resolve(repo)
    missing = [name for name, root in resolution.derived.items() if not root.present]
    raw_present = {name: root.present for name, root in resolution.raw.items()}
    # SiW is the held-out target and is not an input to any derived source tree.
    buildable = raw_present.get("casia_fasd", False) and raw_present.get("msu_mfsd", False)

    return {
        "missing_derived": missing,
        "raw_present": raw_present,
        "can_rebuild": bool(missing) and buildable,
        "nothing_to_do": not missing,
        "blocked": bool(missing) and not buildable,
        "reason": (None if not missing else
                   None if buildable else
                   "the CASIA and MSU raw datasets are required to derive the "
                   "processed and package trees, and at least one is absent"),
    }


def prepare(repo: Path, *, resume: bool = True,
            dry_run: bool = False) -> dict[str, Any]:
    """Build whatever derived tree is missing, in dependency order.

    Returns a report. Raises `PreparationError` only when a build was needed,
    attempted and failed — an absent raw source is reported as a blocked outcome
    rather than an exception, because the caller has a better message for it.
    """
    started = time.time()
    repo = Path(repo).resolve()
    needed = what_is_needed(repo)
    outcomes: list[StepOutcome] = []

    if needed["nothing_to_do"]:
        return _report(started, outcomes, needed,
                       outcome="NOTHING_TO_DO",
                       summary="every derived tree is present")

    if needed["blocked"]:
        return _report(started, outcomes, needed, outcome="BLOCKED",
                       summary=needed["reason"] or "raw data is absent",
                       reason_code=MISSING_RAW_DATA)

    if dry_run:
        return _report(started, outcomes, needed, outcome="WOULD_BUILD",
                       summary=f"would build {', '.join(needed['missing_derived'])}")

    for step in STEPS:
        try:
            outcome = _run_step(repo, step, resume=resume)
        except PreparationError:
            raise
        except Exception as error:                           # noqa: BLE001
            raise PreparationError(
                PREPARATION_FAILED,
                f"derived-data preparation failed at {step}: "
                f"{type(error).__name__}: {error}",
                {"step": step, "completed": [item.name for item in outcomes]}) from error
        outcomes.append(outcome)

    after = what_is_needed(repo)
    if after["missing_derived"]:
        raise PreparationError(
            PREPARATION_FAILED,
            "preparation completed but these derived trees are still absent: "
            + ", ".join(after["missing_derived"]),
            {"still_missing": after["missing_derived"]})

    return _report(started, outcomes, needed, outcome="PREPARED",
                   summary=f"{len([o for o in outcomes if o.action == 'BUILT'])} tree(s) "
                           f"built, {len([o for o in outcomes if o.action == 'REUSED_VALID'])} "
                           "reused")


def _run_step(repo: Path, step: str, *, resume: bool) -> StepOutcome:
    """Dispatch one preparation step to its canonical builder."""
    started = time.time()
    handler = {
        "m2_preprocess": _step_m2,
        "m3a_package": _step_m3a,
        "m3b_priors": _step_m3b,
        "gpat_pairs": _step_pairs,
    }[step]
    outcome = handler(repo, resume=resume)
    outcome.seconds = time.time() - started
    return outcome


def _paths(repo: Path) -> Any:
    from prism_fas.config.models import load_paths

    return load_paths(repo / "configs" / "paths.local.yaml")


def _record_count(repo: Path, dataset: str) -> int:
    """How many records the dataset actually has, via the canonical adapter.

    The same three calls `m2_runner.run` makes to build its record list. Asking
    the adapter is the only way to pass a limit that means "all of them" without
    hard-coding a number that would rot the first time a dataset changed.
    """
    import yaml

    from prism_fas.config.models import load_paths
    from prism_fas.data.m2_runner import DatasetDefinition, adapter_for

    paths = load_paths(repo / "configs" / "paths.local.yaml")
    definition = DatasetDefinition.model_validate(
        yaml.safe_load((repo / "configs" / "data" / f"{dataset}.yaml").read_text(
            encoding="utf-8")))
    return len(adapter_for(definition, getattr(paths.raw_datasets, dataset)).records())


def _step_m2(repo: Path, *, resume: bool) -> StepOutcome:
    """Preprocessing: raw video to canonical frames and manifests."""
    processed = repo / "data" / "processed"
    if processed.is_dir() and any(processed.iterdir()):
        return StepOutcome("m2_preprocess", "REUSED_VALID",
                           "data/processed is present; the canonical M2 runner "
                           "validates per-record hashes and rebuilds nothing")

    from prism_fas.data import m2_runner

    config_path = repo / "configs" / "paths.local.yaml"
    preprocess_path = repo / "configs" / "data" / "preprocess_m2.yaml"
    results: dict[str, Any] = {}
    for dataset in ("casia_fasd", "msu_mfsd"):
        # `limit_records` defaults to 3 in the canonical runner — a smoke default
        # that would silently truncate the scientific corpus. The whole record set
        # is requested explicitly, because L.12 forbids shrinking a scientific
        # input to fit the machine and a default is not a decision.
        records = _record_count(repo, dataset)
        results[dataset] = m2_runner.run(
            dataset, config_path, preprocess_path,
            limit_records=records, dry_run=False, resume=resume, force=False)
    return StepOutcome("m2_preprocess", "BUILT",
                       "preprocessed the CASIA and MSU raw datasets into data/processed",
                       {"per_dataset": {name: (value or {}).get("records")
                                        for name, value in results.items()}})


def _step_m3a(repo: Path, *, resume: bool) -> StepOutcome:
    """The M3A package foundation, plus its validator and lock."""
    from prism_fas.data.package import (build_package, finalize_lock,
                                        load_package_config, validate_package)

    root = repo / "data" / "packages" / "prism_data_v1_m3a"
    if root.is_dir() and (root / "PACKAGE_LOCK.json").is_file():
        report = validate_package(root, require_validated_status=False)
        if report.get("passed"):
            return StepOutcome("m3a_package", "REUSED_VALID",
                               "the M3A package validates; not rebuilt",
                               {"package_root": root.name})

    config = load_package_config(repo / "configs" / "data" / "package_m3a.yaml")
    paths = _paths(repo)
    result = build_package(paths.processed_root, root, config, resume=resume)
    report = validate_package(root)
    if not report.get("passed"):
        raise PreparationError(
            PREPARATION_FAILED,
            "the M3A package was built but did not validate",
            {"validation": report})
    lock = finalize_lock(root, config, report)
    return StepOutcome("m3a_package", "BUILT",
                       "built and validated the M3A source package",
                       {"status": lock.get("status"),
                        "samples": (result or {}).get("samples")})


def _step_m3b(repo: Path, *, resume: bool) -> StepOutcome:
    """Model priors: the pinned towers' features over the packaged frames."""
    from prism_fas.data.package.m3b import build_m3b_package

    source = repo / "data" / "packages" / "prism_data_v1_m3a"
    target = repo / "data" / "packages" / "prism_data_v1_m3b"
    if target.is_dir() and (target / "PACKAGE_LOCK.json").is_file():
        return StepOutcome("m3b_priors", "REUSED_VALID",
                           "the M3B prior package is present and locked")

    from prism_fas.pipeline import portable_paths

    resolution = portable_paths.resolve(repo)
    weight_root = resolution.weights.path if resolution.weights else None
    if weight_root is None:
        raise PreparationError(
            MISSING_RAW_DATA,
            "the pinned model weights could not be resolved; M3B priors are "
            "produced by the frozen towers and there is no substitute")

    result = build_m3b_package(source, target,
                               repo / "configs" / "models" / "model_priors.yaml",
                               weight_root=weight_root, resume=resume)
    return StepOutcome("m3b_priors", "BUILT",
                       "built the M3B model-prior package",
                       {"samples": (result or {}).get("samples")})


def _step_pairs(repo: Path, *, resume: bool) -> StepOutcome:
    """The source-only GPAT pair plan C4 trains against."""
    from prism_fas.synthesis import pair_plan

    output = repo / "data" / "packages" / "gpat_pairs"
    if (output / "PAIR_PLAN_LOCK.json").is_file():
        return StepOutcome("gpat_pairs", "REUSED_VALID",
                           "the GPAT pair plan is present and locked",
                           {"pair_plan_identity": pair_plan.pair_plan_identity(output)})

    package_root = repo / "data" / "packages" / "prism_data_v1_m3a"
    bank_root = repo / "assets" / "recipe_banks" / "c3"
    pair_plan.write_pair_plan(package_root, bank_root, output)
    return StepOutcome("gpat_pairs", "BUILT",
                       "built the source-only GPAT pair plan",
                       {"pair_plan_identity": pair_plan.pair_plan_identity(output)})


def _report(started: float, outcomes: list[StepOutcome], needed: dict[str, Any], *,
            outcome: str, summary: str,
            reason_code: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "summary": summary,
        "reason_code": reason_code,
        "steps": [item.as_dict() for item in outcomes],
        "needed": needed,
        "elapsed_seconds": round(time.time() - started, 2),
        "scientific_eligible": False,
        "meaning": "derived data preparation is deterministic engineering. It "
                   "produces no scientific evidence and selects nothing.",
    }


def write_report(repo: Path, report: dict[str, Any]) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    path = Path(repo) / "reports" / "preflight" / "DERIVED_DATA_PREPARATION.json"
    atomic_write_json(path, report)
    return path


__all__ = ["prepare", "what_is_needed", "write_report", "PreparationError",
           "SCHEMA_VERSION", "STEPS", "MISSING_RAW_DATA", "PREPARATION_FAILED"]
