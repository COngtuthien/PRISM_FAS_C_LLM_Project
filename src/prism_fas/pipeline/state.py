"""``state/PIPELINE_STATE.json`` — current profile, phase, stage and resume cursor.

L.10 requires atomic machine-readable state so a later session or a human can
understand the project without scanning thousands of files. L.11 requires that
an interruption cannot produce an ambiguous completion record, which is why
every write here goes through a temp-file-and-replace rather than a truncating
open: a killed process leaves either the old state or the new one, never half a
file.

The state is a navigation aid. L.10 is explicit that scientific truth remains
the underlying immutable artifact bytes and locks, so nothing in this file is
ever consulted as evidence — only as a cursor.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism_fas.pipeline.profiles import ExecutionProfile
from prism_fas.pipeline.status import assert_not_promoted

PIPELINE_STATE_SCHEMA_VERSION = "prism-pipeline-state-v1"
STATE_DIR = Path("state")
PIPELINE_STATE_FILE = "PIPELINE_STATE.json"


class StateError(RuntimeError):
    """The pipeline state cannot be read, written or trusted."""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so that a crash can never leave a partial file behind.

    ``os.replace`` is atomic on both POSIX and Windows for a same-directory
    rename, so the temp file is created beside the target rather than in the
    system temp directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


@dataclass
class PipelineState:
    """The resume cursor. One per repository, overwritten atomically."""

    profile: str
    phase: str
    stage_id: str | None
    stage_index: int
    completed_stages: list[str] = field(default_factory=list)
    blocked_stages: list[str] = field(default_factory=list)
    last_run_id: str | None = None
    last_updated_utc: str = ""
    outcome: str = "RUNNING"
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_payload(self, profile: ExecutionProfile) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PIPELINE_STATE_SCHEMA_VERSION,
            **profile.stamp(),
            "profile": self.profile,
            "phase": self.phase,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "completed_stages": list(self.completed_stages),
            "blocked_stages": list(self.blocked_stages),
            "last_run_id": self.last_run_id,
            "last_updated_utc": self.last_updated_utc,
            "outcome": self.outcome,
            "notes": list(self.notes),
            "authority": "navigation aid only; scientific truth is the immutable "
                         "artifact bytes and locks (L.10)",
            **self.extra,
        }
        assert_not_promoted(payload)
        return payload


def state_path(repo: Path) -> Path:
    return repo / STATE_DIR / PIPELINE_STATE_FILE


def write_state(repo: Path, state: PipelineState, profile: ExecutionProfile) -> Path:
    path = state_path(repo)
    atomic_write_json(path, state.as_payload(profile))
    return path


def read_state(repo: Path) -> dict[str, Any] | None:
    """Return the stored state, or None when the pipeline has never run.

    A corrupt state file is an error rather than a silent None: L.11 requires
    failing closed on an identity we cannot establish, and an unreadable cursor
    is exactly that.
    """
    path = state_path(repo)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StateError(
            f"{path} is not readable JSON; refusing to resume from an ambiguous cursor "
            f"({error})") from error
    if not isinstance(payload, dict):
        raise StateError(f"{path} does not contain a state mapping")
    return payload


__all__ = ["PIPELINE_STATE_SCHEMA_VERSION", "STATE_DIR", "PIPELINE_STATE_FILE",
           "StateError", "atomic_write_json", "PipelineState", "state_path",
           "write_state", "read_state"]
