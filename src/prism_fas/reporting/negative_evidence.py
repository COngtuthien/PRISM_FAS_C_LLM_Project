"""A machine-readable index of everything that failed, and why it was kept.

Failures are the part of this project most likely to be lost, and the part most
worth keeping. A refused profile, a retained C5 semantic generation failure, a
diverged C7 trial and a crashed GPU run are four different kinds of thing, and
flattening them into "some runs failed" destroys the distinction the paper needs:

* a **SCIENTIFIC_NEGATIVE_RESULT** is a real answer. The candidate whose artifact
  did not survive uint8 quantization, the configuration that diverged inside the
  frozen envelope, the arm that produced fewer accepted samples. These belong in
  Results and Discussion, and deleting one would bias every number computed from
  what remains.
* an **ENGINEERING_FAILURE** is a defect in our code that stopped a run. The
  `Thresholds`-vs-`dict` type mismatch that aborted C6 is the canonical example.
  It says nothing about face anti-spoofing and everything about fail-closed
  methodology, so it belongs in Reproducibility, never in Results.
* a **BLOCKED_PROTOCOL_DECISION** is a question we refused to answer by default.
  BA_sep's probe protocol, C7's search training arm. These belong in Limitations,
  and recording them is what stops a later reader assuming a default was fine.

The classification is the whole value of the index, so it is a closed vocabulary
checked at write time rather than free text. `result_affecting` and
`target_accessed` are recorded per entry because the first decides whether a
finding can change a number and the second is the firewall's own audit trail.

This module holds no scientific constant. Entries are supplied by callers and
seeded from records that already exist in `docs/PROJECT_STATE.md` and in stage
artifacts; nothing here invents a log path for an event whose log was not kept.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "prism-negative-evidence-index-v1"

#: The canonical location. One index for the whole project, not one per stage.
INDEX_PATH = "reports/evidence/NEGATIVE_EVIDENCE_INDEX.json"

#: The closed classification vocabulary. Adding a value is a deliberate act.
SCIENTIFIC_NEGATIVE_RESULT = "SCIENTIFIC_NEGATIVE_RESULT"
ENGINEERING_FAILURE = "ENGINEERING_FAILURE"
BLOCKED_PROTOCOL_DECISION = "BLOCKED_PROTOCOL_DECISION"

CLASSIFICATIONS: tuple[str, ...] = (SCIENTIFIC_NEGATIVE_RESULT, ENGINEERING_FAILURE,
                                    BLOCKED_PROTOCOL_DECISION)

#: Where an entry may legitimately appear in the paper. A closed set for the same
#: reason: "it's in there somewhere" is not a retention policy.
RESULTS = "Results"
DISCUSSION = "Discussion"
LIMITATIONS = "Limitations"
APPENDIX = "Reproducibility/Appendix"

PAPER_SECTIONS: tuple[str, ...] = (RESULTS, DISCUSSION, LIMITATIONS, APPENDIX)

#: The default eligibility per classification. A caller may narrow it; what it may
#: not do is put an engineering defect in Results.
DEFAULT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    SCIENTIFIC_NEGATIVE_RESULT: (RESULTS, DISCUSSION, APPENDIX),
    ENGINEERING_FAILURE: (APPENDIX,),
    BLOCKED_PROTOCOL_DECISION: (LIMITATIONS, APPENDIX),
}

#: An engineering defect may never be presented as a finding about the science.
FORBIDDEN_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    ENGINEERING_FAILURE: (RESULTS, DISCUSSION),
}


class NegativeEvidenceError(ValueError):
    """An entry is not classifiable under the frozen vocabulary."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class NegativeEvidence:
    """One retained failure, blocked decision or negative result."""

    entry_id: str
    stage: str
    substage: str
    classification: str
    reason: str
    #: Where the evidence actually is. Empty when the event predates retention —
    #: recorded as empty rather than invented.
    artifacts: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    occurred_on: str = ""
    result_affecting: bool = False
    target_accessed: bool = False
    paper_eligibility: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise NegativeEvidenceError(
                f"{self.entry_id}: classification {self.classification!r} is not one "
                f"of {CLASSIFICATIONS}")
        eligibility = self.paper_eligibility or DEFAULT_ELIGIBILITY[self.classification]
        unknown = sorted(set(eligibility) - set(PAPER_SECTIONS))
        if unknown:
            raise NegativeEvidenceError(
                f"{self.entry_id}: unknown paper section(s) {unknown}")
        forbidden = sorted(set(eligibility)
                           & set(FORBIDDEN_ELIGIBILITY.get(self.classification, ())))
        if forbidden:
            raise NegativeEvidenceError(
                f"{self.entry_id}: a {self.classification} may not be presented in "
                f"{forbidden}. An implementation defect is reproducibility evidence, "
                "never a finding about face anti-spoofing")
        if self.target_accessed:
            raise NegativeEvidenceError(
                f"{self.entry_id}: declares target_accessed=true. No event recorded so "
                "far touched the held-out target, and an entry claiming one would be a "
                "firewall violation to investigate rather than an index row to write")
        object.__setattr__(self, "paper_eligibility", tuple(eligibility))
        if not self.reason.strip():
            raise NegativeEvidenceError(f"{self.entry_id}: carries no reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "stage": self.stage,
            "substage": self.substage,
            "classification": self.classification,
            "reason": self.reason.strip(),
            "artifacts": list(self.artifacts),
            "logs": list(self.logs),
            "evidence_retained": bool(self.artifacts or self.logs),
            "occurred_on": self.occurred_on,
            "result_affecting": bool(self.result_affecting),
            "target_accessed": bool(self.target_accessed),
            "paper_eligibility": list(self.paper_eligibility),
            "detail": dict(self.detail),
        }


def build_index(entries: Iterable[NegativeEvidence], *,
                generated_at_utc: str | None = None) -> dict[str, Any]:
    """The whole index, ordered by entry id so the file is reproducible."""
    rows = sorted((item.as_dict() for item in entries),
                  key=lambda row: (row["stage"], row["entry_id"]))
    duplicates = sorted({row["entry_id"] for row in rows
                         if [item["entry_id"] for item in rows].count(row["entry_id"]) > 1})
    if duplicates:
        raise NegativeEvidenceError(f"duplicate entry id(s) {duplicates}")

    by_classification = {name: sum(1 for row in rows if row["classification"] == name)
                         for name in CLASSIFICATIONS}
    body = {
        "schema_version": SCHEMA_VERSION,
        "classifications": list(CLASSIFICATIONS),
        "paper_sections": list(PAPER_SECTIONS),
        "entries": rows,
        "entry_count": len(rows),
        "by_classification": by_classification,
        "by_stage": {stage: sum(1 for row in rows if row["stage"] == stage)
                     for stage in sorted({row["stage"] for row in rows})},
        "result_affecting_count": sum(1 for row in rows if row["result_affecting"]),
        "entries_without_retained_evidence": [row["entry_id"] for row in rows
                                              if not row["evidence_retained"]],
        "target_access": 0,
        "retention_policy": (
            "every entry stays addressable for the life of the project. A losing or "
            "failing unit is evidence and is never deleted after a winner exists "
            "(L.6, L.8), and an entry whose log was not kept records an empty log "
            "list rather than a plausible path"),
    }
    body["index_identity"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()
    body["generated_at_utc"] = generated_at_utc or utc()
    return body


def write_index(repo: Path, entries: Sequence[NegativeEvidence], *,
                path: str = INDEX_PATH) -> str:
    from prism_fas.pipeline.state import atomic_write_json

    destination = Path(repo) / path
    atomic_write_json(destination, build_index(entries))
    return path


def read_index(repo: Path, *, path: str = INDEX_PATH) -> dict[str, Any] | None:
    destination = Path(repo) / path
    if not destination.is_file():
        return None
    try:
        return json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def merge(existing: Sequence[dict[str, Any]],
          additions: Sequence[NegativeEvidence]) -> list[NegativeEvidence]:
    """Existing rows plus new ones, with new rows superseding by entry id.

    Superseding rather than appending, because an entry is a description of one
    event: a second row for the same event would double-count it in the paper.
    Removing an entry is not something this function can do.
    """
    kept = {row["entry_id"]: row for row in existing}
    for item in additions:
        kept[item.entry_id] = item.as_dict()
    return [NegativeEvidence(
        entry_id=row["entry_id"], stage=row["stage"], substage=row.get("substage", ""),
        classification=row["classification"], reason=row["reason"],
        artifacts=tuple(row.get("artifacts") or ()),
        logs=tuple(row.get("logs") or ()),
        occurred_on=row.get("occurred_on", ""),
        result_affecting=bool(row.get("result_affecting", False)),
        target_accessed=bool(row.get("target_accessed", False)),
        paper_eligibility=tuple(row.get("paper_eligibility") or ()),
        detail=dict(row.get("detail") or {})) for row in kept.values()]


__all__ = ["SCHEMA_VERSION", "INDEX_PATH", "CLASSIFICATIONS", "PAPER_SECTIONS",
           "SCIENTIFIC_NEGATIVE_RESULT", "ENGINEERING_FAILURE",
           "BLOCKED_PROTOCOL_DECISION", "RESULTS", "DISCUSSION", "LIMITATIONS",
           "APPENDIX", "DEFAULT_ELIGIBILITY", "FORBIDDEN_ELIGIBILITY",
           "NegativeEvidenceError", "NegativeEvidence", "build_index", "write_index",
           "read_index", "merge", "utc"]
