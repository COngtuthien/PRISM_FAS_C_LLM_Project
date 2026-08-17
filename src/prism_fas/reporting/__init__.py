"""Reporting: figures, tables and the final report, from stored evidence only.

The package has one hard rule, and every module restates it: nothing here trains,
loads a dataset or opens a target label. `assemble` walks the artifacts a run
already wrote, hands them to the plot, table and report writers, and records what
it could not find. A missing artifact produces an explicit absence, never a
plausible-looking placeholder.

`generate` is the single entry point the orchestrator calls at the end of a run.
It works under any profile: a rehearsal produces the same structure under
`reports/rehearsal`, clearly banner-marked as not scientific evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism_fas.reporting import complexity, history, plots, report, resources, tables

SCHEMA_VERSION = "prism-reporting-v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def profiled_entries(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Read a profiling artifact that holds either one subject or many.

    C4 and C7 each profile a single model, so their artifact is one flat object.
    C8 runs the matrix and rolls up one entry per arm under ``models``/``runs``.
    Reading only the flat shape would silently report a single arbitrary arm as
    the stage's result, which is exactly the bug this function was added for.
    """
    entries = payload.get(key)
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return [payload] if payload else []


def assemble(repo: Path, *, reports_root: Path, runs_root: Path,
             profile_name: str, execution_intent: str) -> dict[str, Any]:
    """Gather every artifact the writers can use. Absence is recorded, not filled."""
    repo, reports_root, runs_root = Path(repo), Path(reports_root), Path(runs_root)
    missing: list[str] = []

    def require(path: Path, label: str) -> dict[str, Any]:
        payload = _read_json(path)
        if not payload:
            missing.append(label)
        return payload

    c3_lock = require(repo / "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
                      "C3 scientific bank lock")
    lr_record = require(repo / "reports/handoff/LR_ANCHOR_DECISION_RECORD.json",
                        "approved LR decision record")
    environment = _read_json(repo / "state/ENVIRONMENT_MANIFEST.json")
    index = _read_json(repo / "state/MASTER_RUN_INDEX.json")

    rows = index.get("runs", [])
    negatives = [row for row in rows
                 if row.get("status") in ("FAIL", "BLOCKED", "DIVERGED", "INTERRUPTED")]

    history_rows: list[dict[str, Any]] = []
    for path in sorted(runs_root.rglob(history.HISTORY_FILE)):
        history_rows.extend(history.read_history(path))
    if not history_rows:
        missing.append("training history (no train_history.jsonl found)")

    stage_rows: list[dict[str, Any]] = []
    for stage in ("C4", "C5", "C6", "C7", "C8", "C9"):
        payload = _read_json(reports_root / stage.lower()
                             / f"{stage}_{profile_name.upper()}.json")
        if payload:
            stage_rows.append({
                "stage": stage,
                "engineering_status": payload.get("engineering_status"),
                "scientific_status": payload.get("scientific_status"),
                "outcome": payload.get("validate_gate"),
                "checks_run": payload.get("checks_run"),
                "checks_failed": payload.get("checks_failed")})

    # Scan the STAGE directories only. A bare rglob over reports_root also picks
    # up this layer's own output — on Windows the glob is case-insensitive, so
    # `tables/model_complexity.json` matches `*MODEL_COMPLEXITY.json` and the
    # generated table feeds back into the evidence it was generated from.
    stage_dirs = [reports_root / f"c{index}" for index in range(14)]

    def stage_artifacts(suffix: str) -> list[Path]:
        found: list[Path] = []
        for directory in stage_dirs:
            if directory.is_dir():
                found.extend(sorted(directory.glob(f"*{suffix}")))
        return found

    complexity_rows: list[dict[str, Any]] = []
    for path in stage_artifacts("MODEL_COMPLEXITY.json"):
        for entry in profiled_entries(_read_json(path), "models"):
            block = entry.get("complexity") or {}
            complexity_rows.append({
                "model": entry.get("model"),
                "total_parameters": entry.get("total_parameters"),
                "trainable_parameters": entry.get("trainable_parameters"),
                "frozen_parameters": entry.get("frozen_parameters"),
                "parameter_megabytes": entry.get("parameter_megabytes"),
                "macs": block.get("macs"), "flops": block.get("flops"),
                "status": block.get("status"),
                "unsupported_operations": block.get("unsupported_operations")})

    compute_rows: list[dict[str, Any]] = []
    for path in stage_artifacts("COMPUTE_RESOURCES.json"):
        for payload in profiled_entries(_read_json(path), "runs"):
            device = payload.get("device") or {}
            training = payload.get("training") or {}
            plan = payload.get("microbatch_plan") or {}
            memory = (training.get("memory") or {})
            compute_rows.append({
                "run": payload.get("row_id") or path.parent.name,
                "device": device.get("device"),
                "gpu_name": device.get("gpu_name"),
                "effective_batch": plan.get("effective_batch"),
                "physical_microbatch": plan.get("physical_microbatch"),
                "gradient_accumulation_steps": plan.get("gradient_accumulation_steps"),
                "wall_clock_seconds": training.get("wall_clock_seconds"),
                "steps_per_second": training.get("steps_per_second"),
                "samples_per_second": training.get("samples_per_second"),
                "peak_allocated_mb": memory.get("peak_allocated_mb"),
                "peak_reserved_mb": memory.get("peak_reserved_mb")})

    bank_rows = [
        {"arm": arm, "raw_slots": block.get("raw_slots"),
         "eligible": block.get("eligible"), "selected": block.get("selected"),
         "bank_identity": block.get("bank_identity"),
         "coverage_axes": None, "diversity": None}
        for arm, block in sorted((c3_lock.get("arms") or {}).items())]

    try:
        from prism_fas.pipeline.assets import load_manifest

        assets = load_manifest(repo)["items"]
    except Exception:                                        # noqa: BLE001
        assets = []
        missing.append("portable asset manifest")

    return {
        "execution_intent": execution_intent,
        "scientific": execution_intent == "GPU_SCIENTIFIC_FULL",
        "profile": profile_name,
        "identity": {
            "project": "PRISM-FAS-C-LLM",
            "execution_profile": profile_name,
            "execution_intent": execution_intent,
            "master_index_rows": len(rows),
            "reports_root": reports_root.relative_to(repo).as_posix()
            if reports_root.is_relative_to(repo) else str(reports_root),
        },
        "environment": {key: environment.get(key) for key in
                        ("profile_id", "profile_status", "environment_identity",
                         "action", "gpu_family")} if environment else {},
        "assets": assets,
        "c3": {"lock_identity": c3_lock.get("lock_identity"),
               "status": c3_lock.get("status"),
               "arms": ", ".join(sorted((c3_lock.get("arms") or {})))},
        "lr": {"decision_identity": lr_record.get("decision_identity"),
               "status": lr_record.get("status"),
               **{f"{name}_plan": block.get("search_plan_identity")
                  for name, block in (lr_record.get("frozen_search_plans") or {}).items()}},
        "source_stages": stage_rows,
        "target": {},
        "main_results": [],
        "per_seed": [],
        "hypotheses": [],
        "complexity": complexity_rows,
        "compute": compute_rows,
        "locks": [{"lock": "C3_SCIENTIFIC_BANK_LOCK",
                   "identity": c3_lock.get("lock_identity"),
                   "status": c3_lock.get("status")}] if c3_lock else [],
        "negatives": [{key: row.get(key) for key in
                       ("run_id", "stage_id", "status", "execution_profile", "notes")}
                      for row in negatives],
        "firewall": {"target_labels_opened": 0, "target_metrics_computed": 0,
                     "real_target_package_resolved": False,
                     "status": "ARMED"},
        "history": history_rows,
        "banks": {row["arm"]: {"coverage": {}, "diversity": None} for row in bank_rows},
        "recipe_bank_analysis": bank_rows,
        "gate_summaries": {},
        "q_values": {},
        "track_g_seeds": {},
        "track_r_seeds": {},
        "prompthead_seeds": {},
        "frame_vs_video": {},
        "scored": {},
        "ch4": {},
        "missing_evidence": missing,
    }


def generate(repo: Path, *, profile_name: str, execution_intent: str) -> dict[str, Any]:
    """Produce plots, tables, the report and the bundle manifest for one run."""
    from prism_fas.pipeline.state import atomic_write_json

    repo = Path(repo)
    reports_root = repo / "reports" / profile_name
    runs_root = repo / "runs" / profile_name
    evidence = assemble(repo, reports_root=reports_root, runs_root=runs_root,
                        profile_name=profile_name, execution_intent=execution_intent)

    figure_report = plots.generate_all(evidence, reports_root / "plots")
    # Table names and evidence keys are not the same vocabulary — the tables are
    # named for the paper, the evidence for the pipeline — so the mapping is
    # explicit. Leaving it implicit silently emitted empty tables for evidence
    # that was present under a different key.
    table_source = {
        "main_results": evidence.get("main_results"),
        "per_seed_results": evidence.get("per_seed"),
        "hypothesis_tests": evidence.get("hypotheses"),
        "quality_gate": evidence.get("quality_gate"),
        "recipe_bank_analysis": evidence.get("recipe_bank_analysis"),
        "model_complexity": evidence.get("complexity"),
        "compute_efficiency": evidence.get("compute"),
        "source_matrix": evidence.get("source_matrix"),
        "target_results": evidence.get("target_results"),
    }
    table_report = tables.generate_all(
        {name: table_source.get(name) or [] for name in tables.DECLARED_TABLES},
        reports_root / "tables")

    evidence["figures"] = figure_report["written"]
    evidence["tables"] = table_report["tables"]
    report_path = report.write_report(reports_root / "final" / "report.html", evidence,
                                      plots_root=reports_root / "plots")

    bundle = report.build_bundle(repo, reports_root=reports_root, runs_root=runs_root,
                                 extra={"execution_intent": execution_intent,
                                        "profile": profile_name,
                                        "scientific_eligible":
                                            execution_intent == "GPU_SCIENTIFIC_FULL"})
    bundle_path = reports_root / "final" / "FINAL_BUNDLE_MANIFEST.json"
    atomic_write_json(bundle_path, bundle)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile_name,
        "execution_intent": execution_intent,
        "report": report_path.relative_to(repo).as_posix(),
        "bundle": bundle_path.relative_to(repo).as_posix(),
        "plots": figure_report,
        "tables": table_report,
        "missing_evidence": evidence["missing_evidence"],
        "device": resources.device_report(),
    }
    atomic_write_json(reports_root / "final" / "REPORTING_SUMMARY.json", summary)
    return summary


__all__ = ["SCHEMA_VERSION", "complexity", "history", "plots", "report", "resources",
           "tables", "assemble", "generate"]
