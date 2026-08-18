"""M10 target evaluation package CLI.

Builds the additive SiW-Mv2 target evaluation artifact pair on the LOCAL data
factory: raw videos never leave this machine, and only the validated processed
FEATURE package is ever uploaded. The evaluator-only label artifact is written to
a separate tree and is never placed in a training or G7 path.

    inventory   audit the 1700-video declared inventory
    plan        full dry-run; writes nothing
    extract     M2 decode + SCRFD + canonical crop into the isolated namespace
    package     M3A quality priors + M3B parsing/pose/visibility (no identity)
    labels      canonicalize and SEAL the evaluator-only label artifact
    reproduce   the 785-live byte-for-byte reproduction gate
    validate    full package validation
    acceptance  the assembled acceptance report
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import typer
from prism_fas.config.models import load_paths
from prism_fas.data import target_eval as te
from prism_fas.utils.core import atomic_json_write, sha256_file

app = typer.Typer(help="M10 additive SiW-Mv2 target evaluation package", no_args_is_help=True)

LAYOUT = Path("configs/data/siw_mv2_target_v2.yaml")
PATHS = Path("configs/paths.local.yaml")
PREPROCESS = Path("configs/data/preprocess_m2.yaml")
PACKAGE_CONFIG = Path("configs/data/package_m10_target.yaml")
MODEL_CONFIG = Path("configs/models/m3b_priors.yaml")
REPORTS = Path("reports/m10")
FROZEN_PACKAGE = Path("data/processed/prism_data_v1_m3b")
LABEL_ROOT = Path("data/evaluation_only/prism_target_v2_labels")
FEATURE_PACKAGE_ID = "prism_target_eval_v2"
M3A_PACKAGE = Path("data/processed/prism_target_eval_v2_m3a")
FEATURE_PACKAGE = Path("data/processed/prism_target_eval_v2")


def _echo(payload: dict) -> None: typer.echo(json.dumps(payload, default=str))


def _layout() -> te.TargetLayoutV2:
    return te.load_target_layout(LAYOUT)


def _audit(paths, layout) -> te.InventoryAudit:
    return te.audit_inventory(paths.raw_datasets.siw_mv2, layout)


def _profile_root(paths, cfg):
    from prism_fas.data.run_profiles import load_profiles, profile_root
    profiles = load_profiles(Path("configs/data/m2_run_profiles.yaml"))
    profile = profiles[te.TARGET_PROFILE]
    return profile, profile_root(paths.work_root, cfg.preprocessing_version, cfg.config_hash, profile)


@app.command("inventory")
def inventory(report_json: Path = typer.Option(REPORTS / "target_inventory_audit.json", "--report-json"),
              config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
              dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Re-audit the declared 1700-video inventory. Nothing expensive runs first."""
    paths = load_paths(config); layout = _layout()
    audit = _audit(paths, layout)
    report = audit.report()
    if not dry_run: atomic_json_write(Path(report_json), report)
    _echo({**{key: report[key] for key in ("passed", "counts", "by_attack_family",
                                           "unmatched_count", "undeclared_family",
                                           "stem_mismatch_count", "duplicate_id_count",
                                           "count_mismatch", "inventory_identity_sha256")},
           "layout_rules_version": layout.layout_rules_version,
           "adapter_version": layout.adapter_version,
           "written": [] if dry_run else [str(report_json)]})
    if not audit.passed: raise typer.Exit(1)


@app.command("plan")
def plan(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
         preprocess_config: Path = typer.Option(PREPROCESS, "--preprocess-config", exists=True, dir_okay=False),
         resume: bool = typer.Option(True, "--resume/--no-resume")) -> None:
    """Full dry-run before ~6800 frames of processing. Writes nothing."""
    from prism_fas.data.preprocess_m2 import load_m2_config, resolve_detector_path
    paths = load_paths(config); layout = _layout(); cfg = load_m2_config(preprocess_config)
    audit = _audit(paths, layout)
    profile, root = _profile_root(paths, cfg)
    payload = te.build_plan(raw_root=paths.raw_datasets.siw_mv2, layout=layout, audit=audit,
                            preprocess_config=cfg, output_namespace=profile.output_namespace,
                            feature_package_id=FEATURE_PACKAGE_ID, label_root=LABEL_ROOT, resume=resume)
    _echo({**payload, "extract_root": str(root), "m3a_package": str(M3A_PACKAGE),
           "feature_package": str(FEATURE_PACKAGE)})
    if not audit.passed:
        typer.echo(json.dumps({"stop": True, "reason": "the inventory audit does not match the frozen "
                                                       "declaration; refusing to process"}))
        raise typer.Exit(1)


@app.command("extract")
def extract(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
            preprocess_config: Path = typer.Option(PREPROCESS, "--preprocess-config", exists=True, dir_okay=False),
            confirm: bool = typer.Option(False, "--confirm-full-run"),
            limit_records: int | None = typer.Option(None, "--limit-records",
                                                     help="deterministic small-acceptance subset"),
            families: str | None = typer.Option(None, "--families",
                                                help="comma-separated families for a targeted subset"),
            include_live: int = typer.Option(0, "--include-live", help="live videos to include in a subset"),
            chunk: int = typer.Option(25, "--chunk", help="records per resume checkpoint"),
            resume: bool = typer.Option(True, "--resume/--no-resume"),
            dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Decode, detect and crop into the ISOLATED target namespace.

    Resume is per chunk: a chunk that has already finished is skipped, so an
    interrupted 1700-video run continues instead of restarting.
    """
    from prism_fas.data.adapters import adapter_for
    from prism_fas.data.m2_runner import run_preprocessing
    from prism_fas.data.manifests.repository import ManifestRepository
    from prism_fas.data.preprocess_m2 import SCRFDDetector, load_m2_config
    from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
    paths = load_paths(config); layout = _layout(); cfg = load_m2_config(preprocess_config)
    audit = _audit(paths, layout)
    if not audit.passed:
        _echo({"passed": False, "reason": "inventory audit failed", "report": audit.report()}); raise typer.Exit(1)
    records = adapter_for(layout.dataset_definition(), paths.raw_datasets.siw_mv2).inference_records()
    selected = _select(records, audit, families=families, include_live=include_live, limit=limit_records)
    profile, root = _profile_root(paths, cfg)
    if len(selected) == len(records) and not confirm:
        raise typer.BadParameter("the full target build requires --confirm-full-run")
    state_path = root / "state" / "records_done.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))["video_ids"]) if (resume and state_path.is_file()) else set()
    pending = [record for record in selected if record.video_id not in done]
    if dry_run:
        _echo({"status": "dry_run", "selected_records": len(selected), "already_done": len(selected) - len(pending),
               "pending_records": len(pending), "planned_frames": len(pending) * cfg.frames_per_video,
               "output_root": str(root), "resume": resume, "written": []})
        return
    layout_paths = M2OutputLayout.from_root(root)
    detector = SCRFDDetector(resolve_detector_path(cfg.scrfd_model_path), cfg.scrfd_input_size,
                             cfg.detector.get("provider", "CPUExecutionProvider"))
    started = time.time(); totals = {"selected": 0, "successful": 0, "failed": 0, "frames": 0, "crops": 0}
    codes: dict[str, int] = {}
    for start in range(0, len(pending), max(1, chunk)):
        window = pending[start:start + max(1, chunk)]
        context = PreprocessingRunContext(
            project_root=paths.project_root, work_root=paths.work_root, run_profile=te.TARGET_PROFILE,
            output_namespace=profile.output_namespace, output_root=layout_paths.output_root,
            crops_root=layout_paths.crops_root, frames_root=layout_paths.frames_root,
            manifests_root=layout_paths.manifests_root, state_root=layout_paths.state_root,
            reports_root=layout_paths.reports_root, logs_root=layout_paths.logs_root,
            run_id=f"m10-target-eval-v2", dataset="siw_mv2", dataset_role="target",
            preprocessing_version=cfg.preprocessing_version, preprocessing_config_hash=cfg.config_hash,
            detector_model_path=resolve_detector_path(cfg.scrfd_model_path), detector_model_sha256=sha256_file(resolve_detector_path(cfg.scrfd_model_path)),
            detector_input_size=cfg.scrfd_input_size, detector_threshold=cfg.detection_threshold,
            all_records=limit_records is None, record_limit=limit_records, sample_limit=None,
            resume=resume, dry_run=False, partial_full_profile=limit_records is not None,
            command="prism m10 target-package extract")
        result = run_preprocessing(context, window, detector=detector, repository_factory=ManifestRepository)
        totals["selected"] += result.samples_selected; totals["successful"] += result.samples_successful
        totals["failed"] += result.samples_failed; totals["frames"] += result.frames_read
        totals["crops"] += result.crops_written
        for code, count in result.failures_by_code.items(): codes[code] = codes.get(code, 0) + count
        done.update(record.video_id for record in window)
        atomic_json_write(state_path, {"video_ids": sorted(done), "records": len(done),
                                       "output_root": str(root)})
        typer.echo(json.dumps({"progress": {"records_done": len(done), "records_total": len(selected),
                                            "samples_successful": totals["successful"],
                                            "samples_failed": totals["failed"],
                                            "elapsed_seconds": round(time.time() - started, 1)}}))
    _echo({"status": "completed", "records_done": len(done), "records_selected": len(selected),
           **te.frame_accounting(planned=totals["selected"], successful=totals["successful"],
                                 failed=totals["failed"]),
           "frames_read": totals["frames"], "crops_written": totals["crops"],
           "failures_by_code": dict(sorted(codes.items())), "output_root": str(root),
           "elapsed_seconds": round(time.time() - started, 1)})


def _select(records, audit: te.InventoryAudit, *, families: str | None, include_live: int,
            limit: int | None):
    """Deterministic subset selection.

    Uses the EVALUATOR-ONLY inventory to pick a subset that exercises the known
    family/stem mismatch cases; the selection never reaches the feature manifest.
    """
    if families is None and not include_live and limit is None: return records
    by_id = {entry.video_id: entry for entry in audit.entries}
    wanted = {name.strip() for name in (families or "").split(",") if name.strip()}
    live = [record for record in records if by_id[record.video_id].label == "live"][:max(0, include_live)]
    spoof = []
    for name in sorted(wanted):
        matching = [record for record in records
                    if by_id[record.video_id].attack_family == name][:max(1, (limit or 8) // max(1, len(wanted)))]
        spoof.extend(matching)
    chosen = live + spoof
    if not chosen: chosen = records[:limit or 8]
    return sorted({record.video_id: record for record in chosen}.values(), key=lambda r: r.video_id)


@app.command("package")
def package(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
            preprocess_config: Path = typer.Option(PREPROCESS, "--preprocess-config", exists=True, dir_okay=False),
            package_config: Path = typer.Option(PACKAGE_CONFIG, "--package-config", exists=True, dir_okay=False),
            model_config: Path = typer.Option(MODEL_CONFIG, "--model-config", exists=True, dir_okay=False),
            m3a_root: Path = typer.Option(M3A_PACKAGE, "--m3a-root"),
            feature_root: Path = typer.Option(FEATURE_PACKAGE, "--feature-root"),
            device: str | None = typer.Option(None, "--device"),
            resume: bool = typer.Option(True, "--resume/--no-resume"),
            dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Build the quality priors, then the parsing/pose/visibility priors.

    No identity embedding is computed: a target prior never receives one.
    """
    from prism_fas.data.package import build_package, finalize_lock, load_m2_samples, load_package_config, validate_package
    from prism_fas.data.package.m3b import build_m3b_package
    from prism_fas.data.preprocess_m2 import load_m2_config
    paths = load_paths(config); cfg = load_m2_config(preprocess_config)
    package_cfg = load_package_config(package_config)
    _, root = _profile_root(paths, cfg)
    if dry_run:
        samples = load_m2_samples(root)
        _echo({"status": "dry_run", "m2_samples": len(samples), "extract_root": str(root),
               "m3a_root": str(m3a_root), "feature_root": str(feature_root),
               "package_id_prefix": package_cfg.package_id_prefix, "written": []})
        return
    result = build_package(root, Path(m3a_root), package_cfg, resume=resume,
                           progress=lambda stage, done, total: typer.echo(
                               json.dumps({"progress": {"stage": stage, "done": done, "total": total}})))
    pre = validate_package(Path(m3a_root), require_validated_status=False)
    if not pre["passed"]:
        _echo({"stage": "m3a", "passed": False, "errors": pre["errors"][:10]}); raise typer.Exit(1)
    m3a_lock = finalize_lock(Path(m3a_root), pre)
    m3b = build_m3b_package(Path(m3a_root), Path(feature_root), Path(model_config),
                            weight_root=paths.model_cache, resume=resume, device=device,
                            package_id=FEATURE_PACKAGE_ID,
                            progress=lambda payload: typer.echo(json.dumps({"progress": payload})))
    post = validate_package(Path(feature_root), require_validated_status=False, parent_package=Path(m3a_root))
    if not post["passed"]:
        _echo({"stage": "m3b", "passed": False, "errors": post["errors"][:10]}); raise typer.Exit(1)
    feature_lock = finalize_lock(Path(feature_root), post)
    _echo({"status": "completed", "m3a_package_id": m3a_lock["package_id"],
           "m3a_identity": m3a_lock["content_identity_sha256"],
           "feature_package_id": feature_lock["package_id"],
           "feature_identity": feature_lock["content_identity_sha256"],
           "samples": feature_lock["total_samples"], "per_split": feature_lock["per_split_counts"],
           "prior_counts": feature_lock.get("prior_counts"), "device": m3b.get("device"),
           "validation_checks": len(post["checks"]), "validation_errors": len(post["errors"])})


@app.command("labels")
def labels(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
           label_root: Path = typer.Option(LABEL_ROOT, "--label-root"),
           feature_root: Path = typer.Option(FEATURE_PACKAGE, "--feature-root"),
           dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Canonicalize the labels ONCE and seal them in the evaluator-only tree.

    Building and sealing is NOT revealing: nothing here trains, tunes, selects a
    checkpoint, compares variants or computes a target metric.
    `target_labels_revealed` stays false.
    """
    paths = load_paths(config); layout = _layout()
    audit = _audit(paths, layout)
    if not audit.passed:
        _echo({"passed": False, "reason": "inventory audit failed"}); raise typer.Exit(1)
    rows = te.evaluation_label_rows(audit)
    result = te.seal_evaluation_labels(Path(label_root), rows, layout=layout,
                                       feature_package_id=FEATURE_PACKAGE_ID,
                                       inventory_identity=audit.identity(), dry_run=dry_run)
    lock = result["lock"]
    _echo({"status": "sealed" if not dry_run else "dry_run", "rows": result["rows"],
           "counts": lock["counts"], "families": len(lock["by_attack_family"]),
           "label_artifact_identity_sha256": lock["label_artifact_identity_sha256"],
           "label_file_sha256": lock.get("label_file_sha256"),
           "target_label_artifact_built": True, "target_labels_revealed": False,
           "written": result["written"]})


@app.command("reproduce")
def reproduce(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
              frozen_package: Path = typer.Option(FROZEN_PACKAGE, "--frozen-package"),
              feature_root: Path = typer.Option(FEATURE_PACKAGE, "--feature-root"),
              report_json: Path = typer.Option(REPORTS / "target_live_reproduction.json", "--report-json"),
              dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """The primary acceptance gate: 785-live byte-for-byte reproduction."""
    from prism_fas.data.package.manifests import read_manifest
    paths = load_paths(config); layout = _layout()
    audit = _audit(paths, layout)
    live_ids = [entry.video_id for entry in audit.entries if entry.label == "live"]
    rows = read_manifest(Path(feature_root) / "manifests" / "target_test_features.parquet")
    report = te.live_reproduction_audit(frozen_package=Path(frozen_package), new_manifest_rows=rows,
                                        live_video_ids=live_ids)
    if not dry_run: atomic_json_write(Path(report_json), report)
    _echo({**{key: report[key] for key in ("passed", "frozen_live_videos", "frozen_comparable_frames",
                                           "rebuilt_frames_present", "sample_id_mismatches",
                                           "crop_sha256_mismatches", "field_mismatches")},
           "crop_mismatch_examples": report["crop_mismatch_examples"][:3],
           "missing_sample_ids": report["missing_sample_ids"][:3],
           "written": [] if dry_run else [str(report_json)]})
    if not report["passed"]: raise typer.Exit(1)


@app.command("acceptance")
def acceptance(config: Path = typer.Option(PATHS, "--config", exists=True, dir_okay=False),
               preprocess_config: Path = typer.Option(PREPROCESS, "--preprocess-config", exists=True, dir_okay=False),
               feature_root: Path = typer.Option(FEATURE_PACKAGE, "--feature-root"),
               label_root: Path = typer.Option(LABEL_ROOT, "--label-root"),
               output: Path = typer.Option(REPORTS / "TARGET_PACKAGE_ACCEPTANCE.json", "--output"),
               dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Assemble the acceptance report. Contains no detector target metric."""
    from prism_fas.data.package import validate_package
    from prism_fas.data.package.manifests import read_manifest
    from prism_fas.data.preprocess_m2 import load_m2_config
    paths = load_paths(config); layout = _layout(); cfg = load_m2_config(preprocess_config)
    audit = _audit(paths, layout)
    _, root = _profile_root(paths, cfg)
    feature_lock = json.loads((Path(feature_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    label_lock = json.loads((Path(label_root) / "TARGET_LABEL_LOCK.json").read_text(encoding="utf-8"))
    rows = read_manifest(Path(feature_root) / "manifests" / "target_test_features.parquet")
    failures = read_manifest(root / "manifests" / "preprocessing_failures.parquet") if (
        root / "manifests" / "preprocessing_failures.parquet").is_file() else []
    reproduction = json.loads((REPORTS / "target_live_reproduction.json").read_text(encoding="utf-8"))
    validation = validate_package(Path(feature_root))
    privacy = te.assert_features_label_free(rows)
    identity = te.assert_no_target_identity(Path(feature_root))
    planned = len(audit.entries) * cfg.frames_per_video
    accounting = {"planned_nominal": planned, "successful": len(rows), "recorded_failures": len(failures),
                  "reconciled_against_recorded_failures": planned == len(rows) + len(failures)}
    package = te.package_identity(feature_lock=feature_lock, layout=layout, audit_identity=audit.identity(),
                                  reproduction=reproduction,
                                  failure_identity=te.failure_manifest_identity(failures))
    payload = {"target_package_acceptance_schema_version": "m10-target-acceptance-v1",
               "package_id": feature_lock["package_id"],
               "feature_content_identity_sha256": feature_lock["content_identity_sha256"],
               "target_package_identity_sha256": package["target_package_identity_sha256"],
               "label_artifact_identity_sha256": label_lock["label_artifact_identity_sha256"],
               "label_file_sha256": label_lock.get("label_file_sha256"),
               "inventory": audit.report()["counts"], "by_attack_family": audit.by_family(),
               "inventory_audit_passed": audit.passed,
               "frame_accounting": accounting, "crops": len(rows),
               "priors": feature_lock.get("prior_counts"), "shards": len(feature_lock.get("shards") or []),
               "target_identity_embeddings": identity["target_identity_embeddings"],
               "feature_label_leakage": privacy["forbidden_fields_present"],
               "live_reproduction": {key: reproduction[key] for key in
                                     ("passed", "frozen_comparable_frames", "sample_id_mismatches",
                                      "crop_sha256_mismatches")},
               "package_validation": {"passed": validation["passed"],
                                      "checks": len(validation["checks"]),
                                      "errors": len(validation["errors"])},
               "target_labels_revealed": False, "target_label_artifact_built": True,
               "contains_detector_target_metrics": False,
               "package_identity_binding": package}
    payload["passed"] = bool(audit.passed and reproduction["passed"] and validation["passed"]
                             and privacy["forbidden_fields_present"] == 0
                             and identity["target_identity_embeddings"] == 0
                             and accounting["reconciled_against_recorded_failures"])
    if not dry_run: atomic_json_write(Path(output), payload)
    _echo({key: payload[key] for key in ("passed", "package_id", "feature_content_identity_sha256",
                                         "target_package_identity_sha256",
                                         "label_artifact_identity_sha256", "inventory",
                                         "frame_accounting", "target_identity_embeddings",
                                         "feature_label_leakage", "live_reproduction",
                                         "package_validation", "target_labels_revealed")}
          | {"written": [] if dry_run else [str(output)]})
    if not payload["passed"]: raise typer.Exit(1)
