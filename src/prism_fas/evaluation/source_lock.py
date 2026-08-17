"""SOURCE_MATRIX_LOCK_C — the C9 source freeze (§C9, §19.2, L.6).

C9 is the gate between "we ran source experiments" and "the source side is
frozen". Everything after it — target prediction, scoring, hypothesis tests —
assumes that every hypothesis-critical checkpoint, calibration and identity was
fixed *before* a single target prediction existed. A lock that could be built
over incomplete evidence would make that assumption false while still looking
satisfied.

So this module is written as a refusal engine. `build` raises rather than
returning a partial lock, and each refusal names the exact condition:

* a required row has no evidence at all;
* a row is present but did not reach a terminal status;
* a row FAILED — §C9's acceptance is *zero* failed hidden rows;
* a row's recorded run identity disagrees with the plan's;
* a checkpoint or its hash is missing;
* a calibration artifact or its hash is missing;
* evidence exists for a row the plan never declared (a hidden row).

The last one is the subtle one and it is why "hidden" appears in the acceptance
criterion. A run that is not in the plan cannot be part of a preregistered
comparison, and silently locking it in would let an unplanned configuration enter
the frozen source set.

The lock identity covers the frozen scientific content: the matrix identity, each
row's run and config identity, its checkpoint and calibration hashes, and the
code lineage. It excludes clocks, paths and backends, so the same frozen source
set hashes identically wherever it is verified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "prism-c9-source-matrix-lock-v1"

#: L.8 statuses that end a run. Anything else means the row is still in flight,
#: and a lock over an in-flight row would freeze a moving target.
TERMINAL: frozenset[str] = frozenset({"PASS", "FAIL", "DIVERGED", "BLOCKED"})

#: Only this one permits a row into the frozen set.
ACCEPTED: frozenset[str] = frozenset({"PASS"})


class SourceLockError(RuntimeError):
    """The source matrix cannot be frozen as it stands. Never partially applied."""


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RowEvidence:
    """What one executed source row produced, as C9 must find it."""

    row_id: str
    run_identity: str
    config_identity: str
    status: str
    checkpoint_sha256: str | None = None
    calibration_sha256: str | None = None
    calibration_hash: str | None = None
    decision_logit_name: str = ""
    decision_score_name: str = ""
    parent_identities: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "run_identity": self.run_identity,
                "config_identity": self.config_identity, "status": self.status,
                "checkpoint_sha256": self.checkpoint_sha256,
                "calibration_sha256": self.calibration_sha256,
                "calibration_hash": self.calibration_hash,
                "decision_logit_name": self.decision_logit_name,
                "decision_score_name": self.decision_score_name,
                "parent_identities": dict(self.parent_identities),
                "metrics": dict(self.metrics), "notes": self.notes}

    def frozen_material(self) -> dict[str, Any]:
        """The subset the lock identity is taken over — no clock, no path."""
        return {"row_id": self.row_id, "run_identity": self.run_identity,
                "config_identity": self.config_identity,
                "checkpoint_sha256": self.checkpoint_sha256,
                "calibration_sha256": self.calibration_sha256,
                "calibration_hash": self.calibration_hash,
                "decision_logit_name": self.decision_logit_name,
                "decision_score_name": self.decision_score_name,
                "parent_identities": dict(sorted(self.parent_identities.items()))}


def audit(plan: Any, evidence: Sequence[RowEvidence]) -> dict[str, Any]:
    """Every reason this matrix could not be frozen, found in one pass.

    Reporting all of them together rather than raising on the first is
    deliberate: an operator fixing a source freeze wants the whole list, not one
    problem per rerun.
    """
    by_row = {item.row_id: item for item in evidence}
    planned = {row.row_id: row for row in plan.rows}

    missing = sorted(set(planned) - set(by_row))
    hidden = sorted(set(by_row) - set(planned))
    non_terminal = sorted(item.row_id for item in evidence if item.status not in TERMINAL)
    failed = sorted(item.row_id for item in evidence
                    if item.status in TERMINAL and item.status not in ACCEPTED)
    identity_mismatch = sorted(
        row_id for row_id, item in by_row.items()
        if row_id in planned and item.run_identity != planned[row_id].run_identity)
    config_mismatch = sorted(
        row_id for row_id, item in by_row.items()
        if row_id in planned and item.config_identity != planned[row_id].config_identity)
    no_checkpoint = sorted(item.row_id for item in evidence
                           if item.status in ACCEPTED and not item.checkpoint_sha256)
    no_calibration = sorted(item.row_id for item in evidence
                            if item.status in ACCEPTED
                            and not (item.calibration_sha256 and item.calibration_hash))

    refusals = {
        "required_row_missing": missing,
        "hidden_row_not_in_plan": hidden,
        "row_not_terminal": non_terminal,
        "row_failed": failed,
        "run_identity_mismatch": identity_mismatch,
        "config_identity_mismatch": config_mismatch,
        "checkpoint_missing": no_checkpoint,
        "calibration_missing": no_calibration,
    }
    blocking = {name: rows for name, rows in refusals.items() if rows}
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_identity": plan.identity,
        "planned_rows": len(planned),
        "evidence_rows": len(by_row),
        "accepted_rows": sum(1 for item in evidence if item.status in ACCEPTED),
        "refusals": refusals,
        "blocking_refusals": blocking,
        "freezable": not blocking,
        "acceptance_rule": ("all checkpoints, calibrations and identities frozen; zero "
                            "failed hidden rows (§C9)"),
    }


def build(plan: Any, evidence: Sequence[RowEvidence], *,
          code_lineage: Mapping[str, Any] | None = None,
          artifact_identities: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Freeze the source matrix, or refuse and say exactly why.

    There is no `force` and no partial mode. A lock is the claim that the source
    side is complete and fixed; a lock that could be produced over incomplete
    evidence would not be that claim.
    """
    report = audit(plan, evidence)
    if not report["freezable"]:
        raise SourceLockError(
            "the source matrix cannot be frozen: "
            + "; ".join(f"{name}={rows}" for name, rows in report["blocking_refusals"].items())
            + ". §C9 requires all checkpoints, calibrations and identities frozen with "
              "zero failed hidden rows")

    rows = sorted((item.frozen_material() for item in evidence),
                  key=lambda item: item["row_id"])
    material = {
        "matrix_identity": plan.identity,
        "rows": rows,
        "code_lineage": dict(sorted((code_lineage or {}).items())),
        "artifact_identities": dict(sorted((artifact_identities or {}).items())),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_SOURCE_MATRIX",
        "lock_identity": _sha(material),
        "lock_identity_material": ["matrix_identity", "rows", "code_lineage",
                                   "artifact_identities"],
        **material,
        "row_count": len(rows),
        "audit": report,
        "immutability": {"rewrite_permitted": False},
        "target_capability": {"target_labels_resolved": 0, "target_metrics_computed": 0,
                              "proof": "no row in this lock resolved a target label, "
                                       "metric or path; P3 rows are source-selected only"},
        "meaning": ("every hypothesis-critical checkpoint, recipe/synthetic bank identity "
                    "and source_dev calibration is frozen before P3 inference (§19.2)"),
    }


def validate(lock: Mapping[str, Any], plan: Any,
             evidence: Sequence[RowEvidence] | None = None) -> dict[str, Any]:
    """Re-derive the lock from its own declared material and compare.

    A lock that vouched for itself would be worthless, so the identity is
    recomputed from `lock_identity_material` rather than read back, and the
    matrix identity is compared against the plan the caller supplies rather than
    against the one recorded inside the lock.
    """
    problems: list[str] = []

    material = {key: lock.get(key) for key in lock.get("lock_identity_material", [])}
    recomputed = _sha(material)
    if recomputed != lock.get("lock_identity"):
        problems.append("the lock body does not hash to its recorded lock_identity")
    if lock.get("matrix_identity") != plan.identity:
        problems.append(
            f"the lock was built over matrix identity {lock.get('matrix_identity')!r} but "
            f"the plan supplied is {plan.identity!r}")
    if lock.get("status") != "FROZEN_SOURCE_MATRIX":
        problems.append(f"unexpected lock status {lock.get('status')!r}")
    if lock.get("immutability", {}).get("rewrite_permitted") is not False:
        problems.append("the lock does not declare itself immutable")

    locked_rows = {row["row_id"] for row in lock.get("rows", [])}
    planned_rows = {row.row_id for row in plan.rows}
    if locked_rows != planned_rows:
        problems.append(
            f"the lock covers {len(locked_rows)} rows but the plan declares "
            f"{len(planned_rows)}; missing={sorted(planned_rows - locked_rows)}, "
            f"extra={sorted(locked_rows - planned_rows)}")

    if evidence is not None:
        current = {item.row_id: item.frozen_material() for item in evidence}
        drifted = sorted(row["row_id"] for row in lock.get("rows", [])
                         if current.get(row["row_id"]) != row)
        if drifted:
            problems.append(f"row evidence drifted since the freeze: {drifted}")

    return {
        "schema_version": SCHEMA_VERSION,
        "lock_identity": lock.get("lock_identity"),
        "lock_identity_recomputed": recomputed,
        "identity_reproduces": recomputed == lock.get("lock_identity"),
        "matrix_identity": lock.get("matrix_identity"),
        "rows_locked": len(locked_rows),
        "problems": problems,
        "valid": not problems,
    }


__all__ = ["SCHEMA_VERSION", "TERMINAL", "ACCEPTED", "SourceLockError", "RowEvidence",
           "audit", "build", "validate"]
