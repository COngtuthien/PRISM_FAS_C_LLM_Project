"""The bounded one-pass coordinate-search engine (§15.2.2, §15.2.3, L.6).

The engine executes a `SearchPlan` and nothing else. It cannot widen a candidate
set, cannot revisit a coordinate, cannot start a second pass and cannot choose a
winner by any rule other than the plan's frozen selection tuple. Everything it is
*not* allowed to do is therefore absent from its interface rather than guarded by
a convention.

Four behaviours carry most of the contract:

* **One pass, in the declared order.** §15.2.2: "perform ONE pass only and never
  revisit a coordinate". The engine walks `plan.coordinates` once, and a
  completed coordinate is never reopened even if a later coordinate improves the
  objective. That is the whole point — revisiting is how a bounded search turns
  into an unbounded one.
* **Every trial is retained.** L.6 and L.8 forbid winner-only cleanup. Failed,
  NaN and divergent trials stay in the leaderboard; they are ranked after every
  finite valid trial rather than dropped, so "we tried it and it diverged" stays
  answerable.
* **Identity before execution.** L.6 requires a stable `config_id` and canonical
  config hash for every attempted config *before* it runs. They are computed at
  trial construction, which is also what makes resume exact: a resumed run
  matches recorded trials by config hash, not by position.
* **The tie-break is the last word.** After every numeric field of the selection
  tuple, ties break by canonical config SHA-256 ascending. Traversal order,
  thread scheduling and dict ordering cannot reach the result.

The engine knows nothing about GPAT, detectors, metrics or checkpoints. It calls
an `evaluate` callable supplied by the caller and reads the metric names its plan
declares. Keeping it that ignorant is what lets C4 and C7 share one search
implementation without either becoming the other's authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from prism_fas.search.plan import (SearchPlan, canonical_config_sha256, canonical_text,
                                   sha256_text)

SCHEMA_VERSION = "prism-coordinate-search-v1"

#: L.8 run outcomes a trial may carry. A trial is one atomic run.
TRIAL_STATUS: tuple[str, ...] = ("PASS", "FAIL", "DIVERGED", "INTERRUPTED")

#: Statuses whose metrics may rank ahead of the others. §15.2.2: "invalid/NaN/
#: divergent trials are retained and ranked after all finite valid trials".
FINITE_VALID: frozenset[str] = frozenset({"PASS"})

SEARCH_STATE_FILE = "SEARCH_STATE.json"


class SearchError(RuntimeError):
    """The search cannot proceed as the plan declares."""


class SearchInterrupted(Exception):
    """Raised by an evaluator to stop the pass and checkpoint cleanly.

    L.12 requires budget exhaustion to checkpoint and stop cleanly rather than
    truncate the science, and L.11 requires the interruption to leave an
    unambiguous completion record. Raising this is how an evaluator says
    "stop here, and let me resume exactly here later".
    """


class EnvelopeExhausted(SearchError):
    """The whole bounded envelope ran and produced no valid configuration.

    §15.2.2: Claude MUST then STOP with NEEDS_SCIENTIFIC_DECISION and may not
    implement a wider range. Its own exception type so a caller can catch
    exactly this and escalate rather than retry.
    """


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Trial:
    """One attempted configuration, identified before it executes."""

    trial_index: int
    coordinate: str
    value: float
    config: dict[str, Any]
    config_id: str
    config_sha256: str
    search_plan_identity: str

    @classmethod
    def create(cls, *, trial_index: int, coordinate: str, value: float,
               config: Mapping[str, Any], plan_identity: str) -> "Trial":
        sha = canonical_config_sha256(config)
        return cls(trial_index=trial_index, coordinate=coordinate, value=float(value),
                   config=dict(config), config_id=f"{coordinate}@{sha[:12]}",
                   config_sha256=sha, search_plan_identity=plan_identity)

    def as_dict(self) -> dict[str, Any]:
        return {"trial_index": self.trial_index, "coordinate": self.coordinate,
                "value": self.value, "config": dict(self.config),
                "config_id": self.config_id, "config_sha256": self.config_sha256,
                "search_plan_identity": self.search_plan_identity}


@dataclass(frozen=True)
class TrialResult:
    """What one trial produced, including when it produced nothing usable."""

    trial: Trial
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    started_at_utc: str = ""
    finished_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in TRIAL_STATUS:
            raise SearchError(
                f"trial status {self.status!r} is not one of {TRIAL_STATUS}")

    @property
    def finite_valid(self) -> bool:
        """PASS, and every selection metric it reported is a finite number.

        A trial that passed but reported NaN on a ranking metric is not a valid
        ranking candidate: comparing it would let a NaN decide a winner.
        """
        if self.status not in FINITE_VALID:
            return False
        return all(_is_finite(value) for value in self.metrics.values()
                   if isinstance(value, (int, float)) and not isinstance(value, bool))

    def as_dict(self) -> dict[str, Any]:
        return {**self.trial.as_dict(), "status": self.status,
                "metrics": dict(self.metrics), "finite_valid": self.finite_valid,
                "notes": list(self.notes), "artifacts": list(self.artifacts),
                "started_at_utc": self.started_at_utc,
                "finished_at_utc": self.finished_at_utc}


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_value(result: TrialResult, name: str) -> float:
    """One selection-tuple field as a comparable float.

    A missing or non-finite field becomes +inf, which sorts last under
    lexicographic minimization. Booleans are read as 0/1 so a flag field like
    `hard_invariant_failure` ranks False before True without special-casing.
    """
    raw = result.metrics.get(name)
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if raw is None or not _is_finite(raw):
        return math.inf
    return float(raw)


def rank_key(result: TrialResult, selection_tuple: Sequence[str]) -> tuple[Any, ...]:
    """The total order the plan declares, as a sortable key.

    Three tiers, in this order and no other:

    1. finite-valid trials before everything else (§15.2.2);
    2. the plan's selection tuple, minimized lexicographically (§15.4, §15.2.3);
    3. canonical config SHA-256 ascending, the only tie-break (§15.2.2).
    """
    return (0 if result.finite_valid else 1,
            *(_metric_value(result, name) for name in selection_tuple),
            result.trial.config_sha256)


@dataclass
class SearchOutcome:
    """The complete result of one pass: winner, leaderboard and every trial."""

    plan: SearchPlan
    results: list[TrialResult]
    best_config: dict[str, Any]
    completed_coordinates: list[str]
    status: str
    started_at_utc: str
    finished_at_utc: str
    tie_break_trace: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETED"

    @property
    def winner(self) -> TrialResult | None:
        ranked = self.leaderboard()
        if not ranked or not ranked[0].finite_valid:
            return None
        return ranked[0]

    def leaderboard(self) -> list[TrialResult]:
        """Every trial, best first. Losing and failing rows are never dropped."""
        return sorted(self.results, key=lambda item: rank_key(item, self.plan.selection_tuple))

    def as_dict(self) -> dict[str, Any]:
        ranked = self.leaderboard()
        winner = self.winner
        return {
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan.plan_id,
            "milestone": self.plan.milestone,
            "search_plan_identity": self.plan.identity,
            "selection_tuple": list(self.plan.selection_tuple),
            "tie_break": self.plan.tie_break,
            "one_pass": self.plan.one_pass,
            "status": self.status,
            "coordinate_order": list(self.plan.coordinate_order),
            "completed_coordinates": list(self.completed_coordinates),
            "coordinates_skipped": [item.as_dict() for item in self.plan.coordinates
                                    if not item.applicable],
            "trials_declared": self.plan.total_trials,
            "trials_executed": len(self.results),
            "trials_by_status": {status: sum(1 for item in self.results
                                             if item.status == status)
                                 for status in TRIAL_STATUS},
            "finite_valid_trials": sum(1 for item in self.results if item.finite_valid),
            "attempted_config_ids": [item.trial.config_id for item in self.results],
            "leaderboard": [item.as_dict() for item in ranked],
            "winner_config_id": winner.trial.config_id if winner else None,
            "winner_config_sha256": winner.trial.config_sha256 if winner else None,
            "winner_metrics": dict(winner.metrics) if winner else {},
            "best_config": dict(self.best_config),
            "best_config_sha256": canonical_config_sha256(self.best_config),
            "tie_break_trace": list(self.tie_break_trace),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "retention": ("every attempted config is preserved; failed, NaN and divergent "
                          "trials are ranked after all finite valid trials and are never "
                          "deleted (§15.2.2, L.6, L.8)"),
            "expansion_policy": self.plan.expansion_policy,
            "notes": list(self.notes),
        }

    @property
    def outcome_identity(self) -> str:
        """Identity over the search result, excluding clocks and paths."""
        payload = self.as_dict()
        for key in ("started_at_utc", "finished_at_utc"):
            payload.pop(key, None)
        for row in payload["leaderboard"]:
            row.pop("started_at_utc", None)
            row.pop("finished_at_utc", None)
            row.pop("artifacts", None)
        return sha256_text(canonical_text(payload))


# --- resume ------------------------------------------------------------------

def _state_payload(plan: SearchPlan, results: Sequence[TrialResult],
                   best: Mapping[str, Any], completed: Sequence[str],
                   status: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "search_plan_identity": plan.identity,
        "status": status,
        "completed_coordinates": list(completed),
        "best_config": dict(best),
        "results": [item.as_dict() for item in results],
        "updated_utc": _utc(),
    }


def load_state(path: Path, plan: SearchPlan) -> dict[str, Any] | None:
    """Read a prior pass's state, refusing one that belongs to a different plan.

    L.11: if an expected identity changed, fail closed. A search state written
    under a different envelope is not a resume point for this one — reusing it
    would silently mix two search plans in one leaderboard.
    """
    import json

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SearchError(
            f"{path} is not readable JSON; refusing to resume a search from an "
            f"ambiguous state ({error})") from error
    recorded = payload.get("search_plan_identity")
    if recorded != plan.identity:
        raise SearchError(
            f"{path} was written under search plan identity {recorded!r} but this run "
            f"declares {plan.identity!r}; failing closed rather than resuming across two "
            "different frozen envelopes (L.11)")
    return payload


def _rehydrate(payload: Mapping[str, Any], plan: SearchPlan) -> list[TrialResult]:
    results: list[TrialResult] = []
    for row in payload.get("results", []):
        trial = Trial(trial_index=int(row["trial_index"]), coordinate=row["coordinate"],
                      value=float(row["value"]), config=dict(row["config"]),
                      config_id=row["config_id"], config_sha256=row["config_sha256"],
                      search_plan_identity=plan.identity)
        results.append(TrialResult(
            trial=trial, status=row["status"], metrics=dict(row.get("metrics", {})),
            notes=tuple(row.get("notes", ())), artifacts=tuple(row.get("artifacts", ())),
            started_at_utc=row.get("started_at_utc", ""),
            finished_at_utc=row.get("finished_at_utc", "")))
    return results


# --- the engine --------------------------------------------------------------

Evaluator = Callable[[Trial], TrialResult]


def coordinate_search(plan: SearchPlan, evaluate: Evaluator, *,
                      state_path: Path | None = None, resume: bool = False,
                      require_valid_winner: bool = True) -> SearchOutcome:
    """Execute one bounded pass over `plan` and return everything it produced.

    `evaluate` receives a fully identified `Trial` and returns a `TrialResult`.
    It may raise `SearchInterrupted` to stop cleanly; the engine then persists
    state, marks the outcome INTERRUPTED and returns. Any other exception is
    recorded as a FAIL trial and the pass continues, because one broken
    configuration is a finding about that configuration, not a reason to lose
    the results of every other one.

    `resume=True` reuses recorded trials whose config hash matches, so a
    completed coordinate costs nothing on the second run — L.11's requirement
    that a valid completed unit is not recomputed, applied at trial granularity.
    """
    started = _utc()
    results: list[TrialResult] = []
    completed: list[str] = []
    best: dict[str, Any] = dict(plan.base_config)
    notes: list[str] = []
    tie_break_trace: list[dict[str, Any]] = []

    prior_by_sha: dict[str, TrialResult] = {}
    if resume and state_path is not None:
        payload = load_state(state_path, plan)
        if payload is not None:
            prior = _rehydrate(payload, plan)
            prior_by_sha = {item.trial.config_sha256: item for item in prior}
            best = dict(payload.get("best_config") or best)
            notes.append(
                f"resumed from {state_path.name}: {len(prior_by_sha)} recorded trial(s) "
                "are reused by config identity and are not re-executed (L.11)")

    # The anchor itself is the starting point (§15.2.2: "start from the inherited
    # anchor"), so every coordinate not yet searched sits at its anchor value.
    for coordinate in plan.coordinates:
        if coordinate.applicable and coordinate.name not in best:
            best[coordinate.name] = coordinate.anchor

    trial_index = 0
    status = "COMPLETED"

    for coordinate in plan.coordinates:
        if not coordinate.applicable:
            notes.append(f"{coordinate.name}: skipped — {coordinate.skip_reason}")
            continue

        coordinate_results: list[TrialResult] = []
        interrupted = False

        for value in coordinate.candidates:
            trial = Trial.create(
                trial_index=trial_index, coordinate=coordinate.name, value=value,
                config=plan.config_for(coordinate.name, value, best),
                plan_identity=plan.identity)
            trial_index += 1

            recorded = prior_by_sha.get(trial.config_sha256)
            if recorded is not None and recorded.status != "INTERRUPTED":
                reused = TrialResult(
                    trial=trial, status=recorded.status, metrics=dict(recorded.metrics),
                    notes=(*recorded.notes, "reused from recorded state by config identity; "
                                            "not re-executed (L.11)"),
                    artifacts=recorded.artifacts,
                    started_at_utc=recorded.started_at_utc,
                    finished_at_utc=recorded.finished_at_utc)
                results.append(reused)
                coordinate_results.append(reused)
                continue

            trial_started = _utc()
            try:
                result = evaluate(trial)
            except SearchInterrupted as stop:
                results.append(TrialResult(
                    trial=trial, status="INTERRUPTED",
                    notes=(f"interrupted: {stop}",
                           "state was checkpointed; the pass resumes at this exact trial"),
                    started_at_utc=trial_started, finished_at_utc=_utc()))
                interrupted = True
                status = "INTERRUPTED"
                break
            except Exception as error:  # one bad config must not lose the others
                result = TrialResult(
                    trial=trial, status="FAIL",
                    notes=(f"{type(error).__name__}: {error}",),
                    started_at_utc=trial_started, finished_at_utc=_utc())
            results.append(result)
            coordinate_results.append(result)

        if interrupted:
            break

        ranked = sorted(coordinate_results,
                        key=lambda item: rank_key(item, plan.selection_tuple))
        if not ranked:
            continue
        chosen = ranked[0]
        # A coordinate whose every candidate failed keeps the current best rather
        # than adopting a failed value; the failures stay in the leaderboard.
        if chosen.finite_valid:
            best[coordinate.name] = chosen.trial.value
        else:
            notes.append(
                f"{coordinate.name}: no finite valid trial; the coordinate keeps its "
                f"current value {best.get(coordinate.name)!r} and every failed trial is "
                "retained in the leaderboard")
        completed.append(coordinate.name)

        tied = [item for item in ranked if item.finite_valid and
                [_metric_value(item, name) for name in plan.selection_tuple] ==
                [_metric_value(chosen, name) for name in plan.selection_tuple]]
        tie_break_trace.append({
            "coordinate": coordinate.name,
            "candidates_evaluated": [item.trial.value for item in coordinate_results],
            "selected_value": best.get(coordinate.name),
            "selected_config_sha256": chosen.trial.config_sha256,
            "numeric_tie_count": len(tied),
            "decided_by_tie_break": len(tied) > 1,
            "tie_break": plan.tie_break,
            "tied_config_sha256": sorted(item.trial.config_sha256 for item in tied)
            if len(tied) > 1 else [],
        })

        if state_path is not None:
            _write_state(state_path, plan, results, best, completed, "IN_PROGRESS")

    if status == "COMPLETED" and plan.one_pass:
        notes.append("one pass only; no coordinate was revisited (§15.2.2)")

    if state_path is not None:
        _write_state(state_path, plan, results, best, completed, status)

    outcome = SearchOutcome(
        plan=plan, results=results, best_config=best, completed_coordinates=completed,
        status=status, started_at_utc=started, finished_at_utc=_utc(),
        tie_break_trace=tie_break_trace, notes=notes)

    if (require_valid_winner and status == "COMPLETED" and plan.total_trials
            and outcome.winner is None):
        raise EnvelopeExhausted(
            f"the bounded {plan.plan_id!r} envelope executed {len(results)} trial(s) and "
            "produced no finite valid configuration. §15.2.2 requires stopping with "
            "NEEDS_SCIENTIFIC_DECISION rather than widening the search; every attempted "
            "config is preserved in the leaderboard")
    return outcome


def _write_state(path: Path, plan: SearchPlan, results: Sequence[TrialResult],
                 best: Mapping[str, Any], completed: Sequence[str], status: str) -> None:
    from prism_fas.pipeline.state import atomic_write_json

    atomic_write_json(path, _state_payload(plan, results, best, completed, status))


__all__ = ["SCHEMA_VERSION", "TRIAL_STATUS", "FINITE_VALID", "SEARCH_STATE_FILE",
           "SearchError", "SearchInterrupted", "EnvelopeExhausted", "Trial",
           "TrialResult", "rank_key", "SearchOutcome", "load_state",
           "coordinate_search", "Evaluator"]
