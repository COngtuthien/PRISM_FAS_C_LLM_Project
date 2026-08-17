"""Structured training history: enough to redraw every plot without retraining.

A figure that can only be produced by rerunning the job is not evidence, it is a
memory of one. So every training run appends a row per step or epoch to a JSONL
file, and the plotting layer reads only those rows. Nothing downstream is allowed
to call a trainer.

The row schema is deliberately wide — loss components, per-group learning rates,
metrics, invariants, timing, memory — because the expensive part is running the
job, not writing a few hundred bytes per step. A field that was not measured is
absent rather than zero, so a reader can tell "no manifold loss in this variant"
from "the manifold loss was zero".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-training-history-v1"
HISTORY_FILE = "train_history.jsonl"


@dataclass
class HistoryWriter:
    """Append-only structured history for one run.

    Append-only matters for the same reason the master index is: an interrupted
    run must leave the rows it already produced, and a resumed run must add to
    them rather than replace them.
    """

    path: Path
    run_identity: str = ""
    schema_version: str = SCHEMA_VERSION
    rows: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.rows = sum(1 for line in
                            self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def append(self, *, epoch: int, step: int | None = None,
               total_loss: float | None = None,
               losses: dict[str, float] | None = None,
               learning_rates: dict[str, float] | None = None,
               source_train: dict[str, Any] | None = None,
               source_dev: dict[str, Any] | None = None,
               calibration: dict[str, Any] | None = None,
               invariants: dict[str, Any] | None = None,
               selection_tuple: dict[str, Any] | None = None,
               timing: dict[str, Any] | None = None,
               memory: dict[str, Any] | None = None,
               **extra: Any) -> dict[str, Any]:
        """Write one row. Absent measurements are omitted, never zero-filled."""
        row: dict[str, Any] = {"schema_version": self.schema_version,
                               "run_identity": self.run_identity, "epoch": int(epoch)}
        if step is not None:
            row["step"] = int(step)
        if total_loss is not None:
            row["total_loss"] = float(total_loss)
        for key, value in (("losses", losses), ("learning_rates", learning_rates),
                           ("source_train", source_train), ("source_dev", source_dev),
                           ("calibration", calibration), ("invariants", invariants),
                           ("selection_tuple", selection_tuple), ("timing", timing),
                           ("memory", memory)):
            if value:
                row[key] = dict(value)
        row.update({key: value for key, value in extra.items() if value is not None})

        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, default=str) + "\n")
        self.rows += 1
        return row

    @staticmethod
    def group_learning_rates(optimizer: Any) -> dict[str, float]:
        """Every optimizer group's current LR, by name.

        By NAME rather than by index: a variant whose backbone group is empty has
        it omitted entirely, so position 0 means different things in different
        runs and an index would silently mislabel the value.
        """
        rates: dict[str, float] = {}
        for index, group in enumerate(getattr(optimizer, "param_groups", [])):
            rates[str(group.get("name", index))] = float(group.get("lr", 0.0))
        return rates


def read_history(path: Path) -> list[dict[str, Any]]:
    """Every recorded row, in order. A malformed line is skipped, not fatal."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def series(rows: list[dict[str, Any]], key: str, *,
           nested: str | None = None) -> tuple[list[float], list[float]]:
    """An (x, y) series for plotting, dropping rows that lack the field."""
    x: list[float] = []
    y: list[float] = []
    for index, row in enumerate(rows):
        source = row.get(nested, {}) if nested else row
        if not isinstance(source, dict) and nested:
            continue
        value = source.get(key) if nested else row.get(key)
        if value is None:
            continue
        try:
            y.append(float(value))
        except (TypeError, ValueError):
            continue
        x.append(float(row.get("step", row.get("epoch", index))))
    return x, y


__all__ = ["SCHEMA_VERSION", "HISTORY_FILE", "HistoryWriter", "read_history", "series"]
