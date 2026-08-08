"""The M10 report assembler.

Spec Table 54: "reporter: không recompute predictions; chỉ đọc frozen artifacts."
This module reads frozen artifacts and arranges them. It computes no prediction, no
threshold and no calibration, and it never invents a value.

The three properties that matter:

*   **It works with holes.** A FAILED row, a BLOCKED row, a missing optional attack
    metadata column and a `not_applicable` metric all render. Missing optional
    metadata is a rendered absence, not a crash and not a zero.
*   **It never fabricates a target value.** Until real G8 output exists, every
    target section reads `not_yet_scored`. A section is empty because nothing has
    been measured, and it says so.
*   **It refuses an unsupported claim.** A statistical comparison whose either side
    is a single-seed row is refused by the statistics module, and the report prints
    the refusal instead of an interval.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from prism_fas.utils.core import atomic_json_write
from .contracts import (M10_REPORT_SCHEMA_VERSION, M10ContractError, is_not_applicable,
                        may_carry_statistical_claim, not_applicable, stable_identity)

# Spec 16.4, in order. Every section exists in every report, even when empty.
REPORT_SECTIONS = (
    "reproducibility", "dataset_counts", "source_selection_and_calibration",
    "target_frame_metrics", "target_video_metrics", "attack_wise", "region_wise",
    "reliability_and_calibration", "risk_coverage", "confusion_matrix", "threshold_table",
    "compute_and_backend", "baseline_table", "ablations", "statistics", "negative_and_blocked")

NOT_YET_SCORED = "not_yet_scored"


def _empty(reason: str) -> dict[str, Any]:
    return {"status": NOT_YET_SCORED, "reason": reason}


def baseline_table(registry_records: Sequence[Any], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The B00-B08 table. A row with no score shows why, never a blank."""
    rows = []
    for record in registry_records:
        if record.category != "baseline": continue
        scored = scores.get(record.experiment_id)
        rows.append({"experiment_id": record.experiment_id, "family": record.family,
                     "seed": record.seed, "status": record.status,
                     "replication_role": record.replication_role,
                     "source_dev": dict(record.source_dev_metrics),
                     "target": (scored["video"] if scored else
                                _empty(record.blocked_reason or
                                       (record.failure or {}).get("error") or
                                       "no G8 scoring result exists for this row"))})
    return {"rows": sorted(rows, key=lambda row: row["experiment_id"]), "count": len(rows)}


def ablation_table(registry_records: Sequence[Any], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in registry_records:
        if record.category != "ablation": continue
        scored = scores.get(record.experiment_id)
        rows.append({"experiment_id": record.experiment_id, "family": record.family,
                     "variant": record.variant, "seed": record.seed, "status": record.status,
                     "replication_role": record.replication_role,
                     "single_seed_descriptive": not may_carry_statistical_claim(record.replication_role),
                     "target": (scored["video"] if scored else
                                _empty(record.blocked_reason or
                                       (record.failure or {}).get("error") or
                                       "no G8 scoring result exists for this row"))})
    return {"rows": sorted(rows, key=lambda row: row["experiment_id"]), "count": len(rows)}


def negative_and_blocked(registry_records: Sequence[Any],
                         reliability: dict[str, Any] | None = None,
                         statistics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Failures, blocks and non-significant results, in full.

    A negative result is reported, not dropped, not re-tested under a different
    metric and not re-run with another seed until it passes.
    """
    failed = [{"experiment_id": record.experiment_id, "failure": record.failure}
              for record in registry_records if record.status == "FAILED"]
    blocked = [{"experiment_id": record.experiment_id, "reason": record.blocked_reason}
               for record in registry_records if record.status == "BLOCKED"]
    non_significant = []
    for name, entry in ((statistics or {}).get("comparisons") or {}).items():
        if isinstance(entry, dict) and entry.get("significant_at_alpha") is False:
            non_significant.append({"hypothesis": name, "observed_delta": entry.get("observed_delta"),
                                    "ci": [entry.get("ci_low"), entry.get("ci_high")],
                                    "p_value": entry.get("p_value")})
    refused = [{"hypothesis": name, "reason": entry.get("reason")}
               for name, entry in ((statistics or {}).get("comparisons") or {}).items()
               if isinstance(entry, dict) and entry.get("status") == "refused"]
    return {"failed_experiments": failed, "blocked_experiments": blocked,
            "blocked_reliability_tests": (reliability or {}).get("blocked", []),
            "failed_reliability_tests": (reliability or {}).get("failed", []),
            "non_significant_comparisons": non_significant,
            "refused_comparisons": refused,
            "counts": {"failed": len(failed), "blocked": len(blocked),
                       "non_significant": len(non_significant), "refused": len(refused)}}


def assemble(*, plan: dict[str, Any], registry: Any, scores: dict[str, dict[str, Any]] | None = None,
             reliability: dict[str, Any] | None = None, statistics: dict[str, Any] | None = None,
             environment: dict[str, Any] | None = None,
             target_labels_revealed: bool = False) -> dict[str, Any]:
    """Build the full report payload from frozen artifacts only."""
    scores = dict(scores or {})
    records = registry.ordered()
    if scores and not target_labels_revealed:
        raise M10ContractError("a target score cannot exist while target_labels_revealed is false")
    scored_target = bool(scores)
    absent = _empty("no authorized G8 scoring pass has run; target_labels_revealed is false")
    sections: dict[str, Any] = {
        "reproducibility": {
            "m10_matrix_identity": plan["m10_matrix_identity"],
            "config_identity_sha256": plan["config_identity_sha256"],
            "registry_identity": registry.identity(),
            "frozen_inputs": plan["frozen_inputs"],
            "replication_policy": plan["replication_policy"],
            "environment": dict(environment or {}),
            "target_labels_revealed": bool(target_labels_revealed)},
        "dataset_counts": {"planned_rows": plan["summary"]["logical_rows"],
                           "executable_rows": plan["summary"]["executable_rows"],
                           "blocked_rows": plan["summary"]["blocked_rows"],
                           "registry": registry.summary()},
        "source_selection_and_calibration": {
            "rule": plan["rows"][0]["source_selection_rule"] if plan["rows"] else {},
            "selected": [{"experiment_id": record.experiment_id,
                          "best_checkpoint_sha256": record.best_checkpoint_sha256,
                          "source_calibration_sha256": record.source_calibration_sha256,
                          "source_dev_metrics": record.source_dev_metrics}
                         for record in records if record.status == "COMPLETED"],
            "selection_used_target": False},
        "target_frame_metrics": ({name: value["frame"] for name, value in scores.items()}
                                 if scored_target else absent),
        "target_video_metrics": ({name: value["video"] for name, value in scores.items()}
                                 if scored_target else absent),
        "attack_wise": ({name: value["attack_wise"] for name, value in scores.items()}
                        if scored_target else absent),
        "region_wise": ({name: value["region_wise"] for name, value in scores.items()}
                        if scored_target else absent),
        "reliability_and_calibration": {
            "reliability": reliability or _empty("no reliability test has been executed"),
            "target_calibration": ({name: value["video"]["calibration"] for name, value in scores.items()}
                                   if scored_target else absent)},
        "risk_coverage": ({name: value["video"]["risk_coverage"] for name, value in scores.items()}
                          if scored_target else absent),
        "confusion_matrix": ({name: value["video"].get("confusion") for name, value in scores.items()}
                             if scored_target else absent),
        "threshold_table": ({name: value["threshold_table"] for name, value in scores.items()}
                            if scored_target else absent),
        "compute_and_backend": {"by_experiment": {record.experiment_id:
                                                  {"backend": record.backend, **dict(record.compute)}
                                                  for record in records},
                                "note": "offline and online cost are reported separately per Table 58"},
        "baseline_table": baseline_table(records, scores),
        "ablations": ablation_table(records, scores),
        "statistics": statistics or _empty("no statistical comparison has been computed"),
        "negative_and_blocked": negative_and_blocked(records, reliability, statistics)}
    missing = [name for name in REPORT_SECTIONS if name not in sections]
    if missing: raise M10ContractError(f"the report is missing sections {missing}")
    body = {"report_schema_version": M10_REPORT_SCHEMA_VERSION,
            "sections": {name: sections[name] for name in REPORT_SECTIONS},
            "target_labels_revealed": bool(target_labels_revealed),
            "fabricated_target_values": 0,
            "not_claimed": ["SiW-Mv2 performance", "cross-domain superiority",
                            "ablation superiority", "state-of-the-art comparison"]
            if not scored_target else []}
    return {**body, "report_identity": stable_identity(body)}


def audit_no_fabricated_target_values(report: dict[str, Any]) -> dict[str, Any]:
    """Structural proof that an unscored report contains no target number.

    Walks the four target sections and requires every leaf to be either the
    `not_yet_scored` marker or a declared `not_applicable` record. A numeric leaf
    in an unscored report is a fabricated value.
    """
    if report.get("target_labels_revealed"):
        return {"checked": False, "reason": "labels are revealed; target values are expected"}
    offenders = []
    for name in ("target_frame_metrics", "target_video_metrics", "attack_wise", "region_wise",
                 "risk_coverage", "confusion_matrix", "threshold_table"):
        section = report["sections"][name]
        if isinstance(section, dict) and section.get("status") in (NOT_YET_SCORED, "not_applicable"):
            continue
        offenders.append(name)
    if offenders:
        raise M10ContractError(f"an unscored report carries target content in {offenders}")
    return {"checked": True, "sections_verified": 7, "fabricated_target_values": 0}


def write_report(path: Path, payload: dict[str, Any], *, dry_run: bool = False) -> Path:
    target = Path(path)
    if not dry_run: atomic_json_write(target, payload)
    return target


def render_markdown(report: dict[str, Any]) -> str:
    """A compact human rendering. Absences are printed, never skipped."""
    sections = report["sections"]
    lines = ["# M10 report", "",
             f"- report identity `{report['report_identity'][:16]}`",
             f"- matrix identity `{sections['reproducibility']['m10_matrix_identity'][:16]}`",
             f"- target_labels_revealed: **{report['target_labels_revealed']}**", ""]
    for name in REPORT_SECTIONS:
        section = sections[name]
        lines.append(f"## {name.replace('_', ' ')}")
        if isinstance(section, dict) and section.get("status") in (NOT_YET_SCORED, "not_applicable"):
            lines.append(f"_{section['status']}_ — {section.get('reason', '')}")
        elif isinstance(section, dict) and "count" in section:
            lines.append(f"{section['count']} rows")
        else:
            lines.append("recorded")
        lines.append("")
    return "\n".join(lines)
