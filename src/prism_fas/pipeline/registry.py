"""``state/MASTER_RUN_INDEX.json`` — the catalog of every run and artifact (L.10).

L.8 forbids winner-only cleanup: failed, blocked, diverged and interrupted rows
stay addressable from the master index and from the final evidence package. The
index is therefore append-only in behaviour — `record` adds or replaces a row by
its own run identity and never drops one — and the only way a row leaves is if a
human deletes it deliberately.

L.10 calls the index a navigation aid, and that phrasing is load-bearing: the
index is not evidence. Every row points at the artifact bytes that are.

Historical artifacts predating the dual-status model are annotated here rather
than edited in place. C0-C2C acceptance files were written before L.9 existed
and do not carry `execution_profile` or `scientific_eligible`; adding those
fields to a frozen file would change bytes that other identities depend on, so
the index records what they are instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism_fas.pipeline.profiles import ExecutionProfile
from prism_fas.pipeline.state import STATE_DIR, atomic_write_json
from prism_fas.pipeline.status import RUN_OUTCOME, StatusError, assert_not_promoted

MASTER_RUN_INDEX_SCHEMA_VERSION = "prism-master-run-index-v1"
MASTER_RUN_INDEX_FILE = "MASTER_RUN_INDEX.json"


class RegistryError(RuntimeError):
    """The index cannot be read, or a row would misrepresent its evidence."""


@dataclass(frozen=True)
class RunRecord:
    """One row: enough to find the artifact, and enough to know what it is worth.

    L.10 names the required columns — profile, protocol, milestone, method,
    config, seed, status, hashes, parents and paths. A row that omits its
    profile or its eligibility could be read as scientific evidence later, so
    both are mandatory and are checked on construction.
    """

    run_id: str
    stage_id: str
    execution_profile: str
    scientific_eligible: bool
    status: str
    started_at_utc: str
    finished_at_utc: str
    protocol: str | None = None
    method: str | None = None
    config_id: str | None = None
    seed: int | None = None
    config_hash: str | None = None
    content_hash: str | None = None
    parent_identities: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.status not in RUN_OUTCOME:
            raise StatusError(
                f"run status {self.status!r} is not one of {RUN_OUTCOME}")
        assert_not_promoted({"execution_profile": self.execution_profile,
                             "scientific_eligible": self.scientific_eligible})

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "execution_profile": self.execution_profile,
            "scientific_eligible": self.scientific_eligible,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "protocol": self.protocol,
            "method": self.method,
            "config_id": self.config_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "content_hash": self.content_hash,
            "parent_identities": list(self.parent_identities),
            "paths": list(self.paths),
            "notes": self.notes,
            **self.extra,
        }


def index_path(repo: Path) -> Path:
    return repo / STATE_DIR / MASTER_RUN_INDEX_FILE


def read_index(repo: Path) -> dict[str, Any]:
    path = index_path(repo)
    if not path.exists():
        return {"schema_version": MASTER_RUN_INDEX_SCHEMA_VERSION, "runs": [],
                "historical_annotations": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryError(
            f"{path} is not readable JSON; refusing to append to an index whose existing "
            f"rows cannot be preserved ({error})") from error
    payload.setdefault("runs", [])
    payload.setdefault("historical_annotations", [])
    return payload


def record(repo: Path, rows: list[RunRecord], *, profile: ExecutionProfile,
           generated_at_utc: str,
           historical_annotations: list[dict[str, Any]] | None = None) -> Path:
    """Add rows, preserving every row already present.

    Replacement is by ``run_id`` only. A rerun of the same unit updates its own
    row; it can never remove a sibling, which is what keeps losing and failing
    configurations addressable after a winner exists.
    """
    payload = read_index(repo)
    existing: list[dict[str, Any]] = list(payload.get("runs", []))
    by_id = {row.get("run_id"): position for position, row in enumerate(existing)}

    for row in rows:
        serialized = row.as_dict()
        position = by_id.get(row.run_id)
        if position is None:
            by_id[row.run_id] = len(existing)
            existing.append(serialized)
        else:
            existing[position] = serialized

    payload["schema_version"] = MASTER_RUN_INDEX_SCHEMA_VERSION
    payload["generated_at_utc"] = generated_at_utc
    payload["last_writing_profile"] = profile.name
    payload["runs"] = existing
    payload["run_count"] = len(existing)
    payload["authority"] = ("navigation aid only; scientific truth is the immutable "
                            "artifact bytes and locks (L.10)")
    payload["preservation_rule"] = (
        "no winner-only cleanup: PASS, FAIL, DIVERGED, INTERRUPTED and BLOCKED rows are "
        "all retained and remain addressable (L.8)")
    if historical_annotations is not None:
        payload["historical_annotations"] = historical_annotations
    atomic_write_json(index_path(repo), payload)
    return index_path(repo)


def annotate_historical(path: str, *, reason: str,
                        milestone: str) -> dict[str, Any]:
    """Describe a pre-L.9 artifact without touching its bytes.

    The frozen C0-C2C acceptance files carry a single ``result`` field. Editing
    them to add the dual-status fields would change bytes that downstream
    identities were computed over, so the index says what they are and the files
    stay exactly as they were written.
    """
    return {
        "path": path,
        "milestone": milestone,
        "predates": "L.9 dual-status artifact schema",
        "execution_profile": "UNSTAMPED_PRE_V15",
        "scientific_eligible": False,
        "reason": reason,
        "bytes_modified": False,
    }


__all__ = ["MASTER_RUN_INDEX_SCHEMA_VERSION", "MASTER_RUN_INDEX_FILE", "RegistryError",
           "RunRecord", "index_path", "read_index", "record", "annotate_historical"]
