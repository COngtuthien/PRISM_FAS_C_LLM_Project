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
    "reproducibility", "dataset_counts", "target_package", "source_selection_and_calibration",
    "target_frame_metrics", "target_video_metrics", "attack_wise", "region_wise",
    "reliability_and_calibration", "risk_coverage", "confusion_matrix", "threshold_table",
    "compute_and_backend", "backend_parity", "baseline_table", "ablations", "statistics",
    "hypotheses", "known_defects_and_disclosures", "negative_and_blocked")

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
             target_package: dict[str, Any] | None = None,
             backend_parity: dict[str, Any] | None = None,
             hypotheses: dict[str, Any] | None = None,
             disclosures: dict[str, Any] | None = None,
             seed_summaries: dict[str, Any] | None = None,
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
        "target_package": target_package or _empty("no frozen target package record was supplied"),
        "backend_parity": backend_parity or _empty("no backend parity evidence was supplied"),
        "baseline_table": baseline_table(records, scores),
        "ablations": ablation_table(records, scores),
        "statistics": statistics or _empty("no statistical comparison has been computed"),
        "hypotheses": hypotheses or _empty("no hypothesis outcome has been adjudicated"),
        "known_defects_and_disclosures": disclosures or _empty("no disclosure record was supplied"),
        "negative_and_blocked": negative_and_blocked(records, reliability, statistics)}
    if seed_summaries: sections["dataset_counts"]["by_row_seed_summary"] = seed_summaries
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


def _cell(value: Any, digits: int = 5) -> str:
    """Render one metric. `not_applicable` prints its status, never a blank or a zero."""
    from .contracts import is_not_applicable
    if value is None: return "&mdash;"
    if is_not_applicable(value): return "<span class='na'>not_applicable</span>"
    if isinstance(value, bool): return "yes" if value else "no"
    if isinstance(value, float): return f"{value:.{digits}f}"
    return str(value)


def _escape(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_html(report: dict[str, Any], summary: dict[str, Any] | None = None) -> str:
    """A self-contained HTML rendering of the frozen report.

    It recomputes nothing. Every number it shows is read out of the assembled
    report or the summary, and a metric that could not be computed prints
    `not_applicable` with its reason rather than a blank cell.
    """
    import json as _json
    sections = report["sections"]
    summary = summary or {}
    revealed = bool(report.get("target_labels_revealed"))
    out: list[str] = [
        "<title>PRISM-FAS-B — M10 blind target evaluation</title>",
        "<style>",
        ":root{color-scheme:light dark}",
        "body{font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:2rem;max-width:1200px}",
        "h1{font-size:1.7rem;margin:0 0 .2rem}h2{font-size:1.15rem;margin:2.2rem 0 .6rem;"
        "border-bottom:1px solid #8884;padding-bottom:.3rem}",
        "table{border-collapse:collapse;width:100%;font-size:.86rem}",
        "th,td{border:1px solid #8884;padding:.32rem .5rem;text-align:right}",
        "th:first-child,td:first-child{text-align:left;white-space:nowrap}",
        "th{background:#8881;font-weight:600}",
        ".wrap{overflow-x:auto;margin:.4rem 0 1rem}",
        ".na{color:#a66;font-style:italic}.k{color:#888;font-family:ui-monospace,monospace;font-size:.8rem}",
        ".pill{display:inline-block;padding:.1rem .5rem;border:1px solid #8886;border-radius:1rem;"
        "font-size:.78rem;margin-right:.3rem}",
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:.15rem .9rem;margin:.4rem 0}",
        "dt{color:#888}dd{margin:0;font-family:ui-monospace,monospace;font-size:.82rem;word-break:break-all}",
        "pre{background:#8881;padding:.7rem;overflow-x:auto;font-size:.78rem;border-radius:4px}",
        "</style>",
        "<h1>PRISM-FAS-B &mdash; M10 blind target evaluation</h1>",
        f"<p class='k'>report identity {report['report_identity']}<br>"
        f"target_labels_revealed: <b>{revealed}</b></p>"]

    repro = sections["reproducibility"]
    out.append("<h2>1. Reproducibility</h2><dl>")
    for key in ("m10_matrix_identity", "config_identity_sha256", "registry_identity"):
        out.append(f"<dt>{key}</dt><dd>{_escape(repro.get(key))}</dd>")
    for key in ("source_matrix_lock_identity", "target_prediction_lockset_identity",
                "target_feature_package_identity", "target_label_reveal_identity", "summary_identity"):
        if summary.get(key): out.append(f"<dt>{key}</dt><dd>{_escape(summary[key])}</dd>")
    out.append("</dl>")

    package = sections.get("target_package") or {}
    if package.get("status") != NOT_YET_SCORED:
        out.append("<h2>2. Frozen target evaluation package</h2><dl>")
        for key, value in sorted(package.items()):
            out.append(f"<dt>{_escape(key)}</dt><dd>{_escape(value)}</dd>")
        out.append("</dl>")

    if revealed and summary.get("per_seed"):
        out.append("<h2>3. Target video metrics &mdash; every scientific run</h2>")
        out.append("<div class='wrap'><table><tr><th>experiment</th><th>threshold</th><th>ACER</th>"
                   "<th>APCER</th><th>BPCER</th><th>HTER</th><th>ROC-AUC</th><th>EER</th></tr>")
        for name, block in sorted(summary["per_seed"].items()):
            video = block["video"]
            out.append("<tr><td>{}</td><td>{}</td>{}</tr>".format(
                _escape(name), _cell(block.get("threshold")),
                "".join(f"<td>{_cell(video.get(key))}</td>"
                        for key in ("acer", "apcer", "bpcer", "hter", "roc_auc", "eer"))))
        out.append("</table></div>")

        out.append("<h2>4. Mean &plusmn; std over seeds</h2>")
        out.append("<div class='wrap'><table><tr><th>row</th><th>seeds</th><th>role</th>"
                   "<th>video ACER</th><th>video ROC-AUC</th><th>video HTER</th>"
                   "<th>claim-bearing</th></tr>")
        for row, block in sorted((summary.get("by_row") or {}).items()):
            def spread(metric: str, block: dict[str, Any] = block) -> str:
                value = block["video"].get(metric)
                if not isinstance(value, dict): return _cell(value)
                if value.get("status") == "not_applicable": return "<span class='na'>not_applicable</span>"
                if "mean" not in value: return "&mdash;"
                return f"{value['mean']:.5f} &plusmn; {value['std']:.5f}"
            out.append(f"<tr><td>{_escape(row)}</td><td>{block['n_seeds']}</td>"
                       f"<td>{_escape(block['replication_role'])}</td>"
                       f"<td>{spread('acer')}</td><td>{spread('roc_auc')}</td>"
                       f"<td>{spread('hter')}</td>"
                       f"<td>{'yes' if block['may_carry_statistical_claim'] else 'no'}</td></tr>")
        out.append("</table></div>")

    hypotheses = sections.get("hypotheses") or {}
    if hypotheses.get("status") != NOT_YET_SCORED:
        out.append("<h2>5. Predeclared hypotheses</h2>")
        out.append("<div class='wrap'><table><tr><th>H</th><th>treatment</th><th>control</th>"
                   "<th>&Delta;ACER</th><th>95% CI</th><th>p raw</th><th>p Holm</th>"
                   "<th>outcome</th></tr>")
        for name, block in sorted(hypotheses.items()):
            ci = block.get("ci95") or [None, None]
            out.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                       "<td>{}</td><td><b>{}</b></td></tr>".format(
                           _escape(name), _escape(block.get("treatment")), _escape(block.get("control")),
                           _cell(block.get("observed_delta_acer")),
                           "&mdash;" if ci[0] is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]",
                           _cell(block.get("p_value_raw"), 4), _cell(block.get("p_value_holm_adjusted"), 4),
                           _escape(block.get("outcome"))))
        out.append("</table></div>")

    if revealed and isinstance(sections.get("attack_wise"), dict) \
            and sections["attack_wise"].get("status") != NOT_YET_SCORED:
        families = sorted({family for block in sections["attack_wise"].values()
                           if isinstance(block, dict) and block.get("by_family")
                           for family in block["by_family"]})
        if families:
            out.append("<h2>6. Attack-wise APCER (post-hoc, never used in tuning)</h2>")
            out.append("<div class='wrap'><table><tr><th>experiment</th>"
                       + "".join(f"<th>{_escape(name)}</th>" for name in families) + "</tr>")
            for name, block in sorted(sections["attack_wise"].items()):
                if not isinstance(block, dict) or not block.get("by_family"): continue
                out.append(f"<tr><td>{_escape(name)}</td>" + "".join(
                    f"<td>{_cell((block['by_family'].get(family) or {}).get('apcer'))}</td>"
                    for family in families) + "</tr>")
            out.append("</table></div>")

    parity = sections.get("backend_parity") or {}
    if parity.get("status") != NOT_YET_SCORED:
        out.append("<h2>7. Backend parity (H6 &mdash; parity, not superiority)</h2>")
        out.append(f"<pre>{_escape(_json.dumps(parity, indent=1)[:6000])}</pre>")

    disclosures = sections.get("known_defects_and_disclosures") or {}
    if disclosures.get("status") != NOT_YET_SCORED:
        out.append("<h2>8. Disclosures and known defects</h2>")
        for key, block in sorted(disclosures.items()):
            out.append(f"<h3 class='k'>{_escape(key)}</h3>"
                       f"<pre>{_escape(_json.dumps(block, indent=1))}</pre>")

    reliability = (sections.get("reliability_and_calibration") or {}).get("reliability") or {}
    if reliability.get("tests"):
        out.append("<h2>9. Reliability and shortcut tests</h2>")
        out.append("<div class='wrap'><table><tr><th>test</th><th>status</th><th>population</th>"
                   "<th>result / reason</th></tr>")
        for test in reliability["tests"]:
            detail = test.get("blocked_reason") or _json.dumps(test.get("result") or {})
            out.append(f"<tr><td>{_escape(test['test_id'])}</td><td>{_escape(test['status'])}</td>"
                       f"<td style='text-align:left'>{_escape(test['population'])}</td>"
                       f"<td style='text-align:left'>{_escape(detail[:400])}</td></tr>")
        out.append("</table></div>")

    negatives = sections["negative_and_blocked"]
    out.append("<h2>10. Negative results, failures and blocked rows</h2>")
    out.append(f"<pre>{_escape(_json.dumps(negatives, indent=1)[:8000])}</pre>")

    out.append("<h2>11. What this report does not claim</h2><p>"
               + "".join(f"<span class='pill'>{_escape(item)}</span>"
                         for item in (summary.get("not_claimed") or report.get("not_claimed")
                                      or ["state-of-the-art", "first method"]))
               + "</p>")
    return "\n".join(out)


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
