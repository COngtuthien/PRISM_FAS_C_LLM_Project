"""Machine-readable and paper-facing tables, derived from stored artifacts only.

Every cell traces to a canonical artifact. Nothing is typed in, nothing is
rounded into a claim it does not support, and a table with no evidence behind it
is emitted with a header and zero rows rather than omitted — an absent file and
an empty result look identical to a reader otherwise.

Each table is written twice, as CSV for a spreadsheet and JSON for a program, so
the paper path and the audit path read the same numbers.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "prism-tables-v1"

#: The tables the closure contract requires, with the columns each carries.
DECLARED_TABLES: dict[str, tuple[str, ...]] = {
    "main_results": ("experiment_id", "track", "arm", "protocol", "seeds",
                     "video_ACER_mean", "video_ACER_std", "video_BPCER_mean",
                     "APCER_mean", "roc_auc_mean", "eer_mean", "status"),
    "per_seed_results": ("experiment_id", "track", "arm", "protocol", "seed",
                         "video_ACER", "video_BPCER", "APCER", "roc_auc", "eer",
                         "nll", "ece", "status"),
    "hypothesis_tests": ("hypothesis", "comparison", "effect", "ci_low", "ci_high",
                         "p_value", "holm_adjusted_p", "reject_null", "seeds",
                         "statistical_claim_allowed"),
    "quality_gate": ("arm", "candidates", "accepted", "rejected", "acceptance_rate",
                     "q_min", "q_median", "q_mean", "q_max", "profile", "failed_gates"),
    "recipe_bank_analysis": ("arm", "raw_slots", "eligible", "selected",
                             "bank_identity", "coverage_axes", "diversity"),
    "model_complexity": ("model", "total_parameters", "trainable_parameters",
                         "frozen_parameters", "parameter_megabytes", "macs", "flops",
                         "status", "unsupported_operations"),
    "compute_efficiency": ("run", "device", "gpu_name", "effective_batch",
                           "physical_microbatch", "gradient_accumulation_steps",
                           "wall_clock_seconds", "steps_per_second",
                           "samples_per_second", "peak_allocated_mb",
                           "peak_reserved_mb"),
    "source_matrix": ("row_id", "experiment_id", "track", "arm", "protocol", "seed",
                      "config_identity", "run_identity", "status",
                      "checkpoint_sha256", "calibration_hash"),
    "target_results": ("experiment_id", "track", "arm", "seed", "videos",
                       "video_ACER", "APCER", "BPCER", "roc_auc", "eer",
                       "threshold", "prediction_lock_identity"),
}


def write_table(out: Path, name: str, rows: Sequence[dict[str, Any]], *,
                columns: Sequence[str] | None = None) -> dict[str, Any]:
    """One table as CSV and JSON. An empty table is still written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(columns or DECLARED_TABLES.get(name)
                  or (sorted({key for row in rows for key in row}) if rows else []))

    csv_path = out / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _flatten(row.get(field)) for field in fields})

    json_path = out / f"{name}.json"
    json_path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "table": name,
                    "columns": fields, "row_count": len(rows),
                    "rows": [dict(row) for row in rows],
                    "source": "derived from canonical stored artifacts; no value is "
                              "hand-entered"},
                   indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return {"table": name, "rows": len(rows), "columns": fields,
            "csv": csv_path.name, "json": json_path.name,
            "empty_reason": "" if rows else "no evidence exists for this table yet"}


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def generate_all(evidence: dict[str, Any], out: Path) -> dict[str, Any]:
    """Write every declared table from whatever evidence exists.

    `evidence` maps a table name to its rows. A name with no rows still produces
    a header-only CSV, so the reader can see the table was expected and is empty
    rather than wondering whether it was forgotten.
    """
    out = Path(out)
    written: list[dict[str, Any]] = []
    for name in DECLARED_TABLES:
        rows = list(evidence.get(name) or [])
        written.append(write_table(out, name, rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": out.as_posix(),
        "tables": written,
        "table_count": len(written),
        "populated": [item["table"] for item in written if item["rows"]],
        "empty": [item["table"] for item in written if not item["rows"]],
        "note": "an empty table is written with its header so an absent artifact is "
                "distinguishable from an absent table",
    }


__all__ = ["SCHEMA_VERSION", "DECLARED_TABLES", "write_table", "generate_all"]
