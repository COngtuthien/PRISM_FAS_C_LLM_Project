"""M10 CLI: matrix, registry, G7 prediction, prediction lock, G8 scoring,
statistics, reliability, report and acceptance.

Every writing command supports `--dry-run`. A dry run writes nothing, launches no
Modal job and opens no target label; it reports exactly what it would have done.
"""
from __future__ import annotations
import json
from pathlib import Path
import typer
from prism_fas.utils.core import atomic_json_write
from prism_fas.evaluation import bootstrap, experiment_matrix, reliability, report, scoring
from prism_fas.evaluation.contracts import M10ContractError
from prism_fas.evaluation.experiment_registry import (ExperimentRegistry, source_matrix_lock,
                                                      validate_source_matrix_lock)
from prism_fas.evaluation.firewall import TargetLabelFirewall, load_firewall_config
from prism_fas.evaluation import target_prediction as g7

app = typer.Typer(help="M10 experiment matrix and blind target evaluation", no_args_is_help=True)
matrix_app = typer.Typer(help="Deterministic Table 59/60 experiment matrix")
prediction_app = typer.Typer(help="G7 label-free target prediction and its lock")
app.add_typer(matrix_app, name="matrix")
app.add_typer(prediction_app, name="prediction")
from prism_fas.cli.m10_target import app as target_package_app
app.add_typer(target_package_app, name="target-package")

DEFAULT_MATRIX = Path("configs/experiments/m10_matrix.yaml")
DEFAULT_EVALUATION = Path("configs/evaluation/m10_target.yaml")
DEFAULT_REPORTS = Path("reports/m10")


def _firewall(evaluation_config: Path) -> TargetLabelFirewall:
    return TargetLabelFirewall(load_firewall_config(Path(evaluation_config)), project_root=Path.cwd())


def _echo(payload: dict) -> None: typer.echo(json.dumps(payload, default=str))


# --- matrix ------------------------------------------------------------------
@matrix_app.command("plan")
def matrix_plan(config: Path = typer.Option(DEFAULT_MATRIX, "--config", exists=True, dir_okay=False),
                output: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--output"),
                repeat: int = typer.Option(2, "--repeat", help="the contract requires two agreeing passes"),
                dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Materialize the matrix twice and require an identical `m10_matrix_identity`."""
    plans = [experiment_matrix.build_plan(Path(config)) for _ in range(max(1, repeat))]
    agreement = experiment_matrix.plans_agree(plans[0], plans[-1]) if len(plans) > 1 else {
        "identical": True, "m10_matrix_identity": plans[0]["m10_matrix_identity"]}
    if not agreement["identical"]:
        _echo({"passed": False, "determinism": agreement, "written": []}); raise typer.Exit(1)
    if not dry_run: atomic_json_write(Path(output), plans[0])
    _echo({"passed": True, "m10_matrix_identity": plans[0]["m10_matrix_identity"],
           "determinism_passes": len(plans), **plans[0]["summary"],
           "written": [] if dry_run else [str(output)]})


@matrix_app.command("validate")
def matrix_validate(plan: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--plan",
                                              exists=True, dir_okay=False),
                    config: Path = typer.Option(DEFAULT_MATRIX, "--config", exists=True, dir_okay=False),
                    evaluation_config: Path = typer.Option(DEFAULT_EVALUATION, "--evaluation-config",
                                                           exists=True, dir_okay=False)) -> None:
    """Re-derive the plan from the config and check the identity plus the firewall."""
    import yaml
    stored = json.loads(Path(plan).read_text(encoding="utf-8"))
    rebuilt = experiment_matrix.build_plan(Path(config))
    firewall = _firewall(evaluation_config)
    matrix_payload = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    checks = {"identity_matches": stored["m10_matrix_identity"] == rebuilt["m10_matrix_identity"],
              "rows_match": stored["rows"] == rebuilt["rows"],
              "no_target_paths_in_matrix_config": bool(
                  firewall.assert_no_target_paths(matrix_payload, where="m10_matrix.yaml")),
              "no_attack_taxonomy_in_matrix_config": bool(
                  firewall.assert_no_attack_taxonomy(matrix_payload, where="m10_matrix.yaml")),
              "blocked_rows_carry_reasons": all(row.get("blocked_reason")
                                                for row in stored["rows"] if row["status"] == "BLOCKED")}
    _echo({"passed": all(checks.values()), "checks": checks,
           "m10_matrix_identity": stored["m10_matrix_identity"], "summary": stored["summary"],
           "firewall": firewall.report()})
    if not all(checks.values()): raise typer.Exit(1)


@matrix_app.command("audit")
def matrix_audit(config: Path = typer.Option(DEFAULT_MATRIX, "--config", exists=True, dir_okay=False),
                 output: Path = typer.Option(DEFAULT_REPORTS / "M10_IMPLEMENTABILITY.json", "--output"),
                 dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Instantiate and step EVERY executable row before any GPU hour is spent.

    Each unique scientific configuration is built as the real detector, given a real
    batch, run forward and backward through one optimizer step, and checked against
    what it declares. A row that cannot be materialized is reported, never repaired
    into a neighbouring configuration.
    """
    from prism_fas.evaluation.variant_audit import audit_matrix, write_audit
    plan = experiment_matrix.build_plan(Path(config))
    report = audit_matrix(plan)
    if not dry_run: write_audit(Path(output), report)
    _echo({"passed": report["all_executable_rows_implementable"],
           "m10_matrix_identity": report["m10_matrix_identity"],
           "audited_rows": report["audited_rows"],
           "implementable_rows": report["implementable_rows"],
           "unique_variant_configs": report["unique_variant_configs"],
           "unique_scientific_configs_covered": report["unique_scientific_configs_covered"],
           "not_implementable": report["not_implementable"],
           "blocked_rows": [row["experiment_id"] for row in report["blocked_rows"]],
           "written": [] if dry_run else [str(output)]})
    if not report["all_executable_rows_implementable"]: raise typer.Exit(1)


@app.command("parity")
def parity_command(
        experiment_id: str = typer.Option("A09-backend-pc_bounded_parity-s20260806", "--experiment-id"),
        against: str = typer.Option("B08-s20260806", "--against"),
        remote: Path = typer.Option(..., "--remote", exists=True, dir_okay=False,
                                    help="the m10_parity_probe payload from Modal"),
        package_root: Path = typer.Option(..., "--package-root", exists=True, file_okay=False),
        bank_root: Path = typer.Option(..., "--bank-root", exists=True, file_okay=False),
        weight_root: Path = typer.Option(..., "--weight-root", exists=True, file_okay=False),
        steps: int = typer.Option(5, "--steps"),
        shared_checkpoint: Path = typer.Option(..., "--shared-checkpoint", exists=True,
                                               dir_okay=False,
                                               help="the frozen checkpoint BOTH backends start from"),
        output: Path = typer.Option(DEFAULT_REPORTS / "A09_BACKEND_PARITY.json", "--output"),
        dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """A09 — the LOCAL half of the bounded step-parity protocol, and the comparison.

    Runs the declared number of optimizer steps on this host at the B08 configuration
    and seed, then compares against the Modal probe under the M6 tolerances. It is
    parity evidence: it trains nothing to completion, selects no checkpoint and
    makes no superiority claim.
    """
    import torch
    from dataclasses import replace
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.trainer import M9Trainer
    from prism_fas.detector.variant import variant_from_row
    from prism_fas.evaluation.backend_parity import compare_backends, probe_payload, write_parity

    project = Path(__file__).resolve().parents[3]
    plan = experiment_matrix.build_plan(DEFAULT_MATRIX)
    rows = {str(row["experiment_id"]): row for row in plan["rows"]}
    parity_row, reference_row = rows[experiment_id], rows[against]
    # The parity row and its reference must be the SAME science, or this compares
    # two experiments rather than two backends.
    if parity_row["scientific_config_hash"] != reference_row["scientific_config_hash"]:
        _echo({"passed": False, "reason": "the parity row and its reference are not the same "
                                          "scientific configuration"})
        raise typer.Exit(1)
    variant = variant_from_row(parity_row)
    configs = load_m9_configs(project / "configs/models/m9_detector.yaml",
                              project / "configs/train/m9_reference.yaml", variant)
    # `configs/cloud/modal_m6.yaml` declares `parity: precision: fp32`, so BOTH halves
    # set amp=False. It also keeps the two config hashes equal: `amp` is a config
    # field, so overriding it on one side only would report a scientific-identity
    # mismatch that is really just the probe disagreeing with itself.
    training_config = replace(configs["training_config"], run_id=f"m10_parity_local/{experiment_id}",
                              seed=int(reference_row["seed"]), amp=False)
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        trainer = M9Trainer(
            config=training_config, detector_config=configs["detector_config"],
            package_root=Path(package_root), bank_root=Path(bank_root),
            recipe_bank_root=project / "assets/recipe_banks/prism_recipe_bank_m7_v1",
            run_root=Path(scratch) / "run", cache_root=Path(scratch) / "cache",
            weight_root=Path(weight_root),
            loader_config_path=project / "configs/data/loader_m4.yaml",
            device="cuda" if torch.cuda.is_available() else "cpu")
            # `text_cache_path` is deliberately not forced: the resolver searches the
            # local cache layout AND the Modal volume layout, and it verifies the
            # frozen cache's identity either way. Forcing one spelling would make the
            # local half fail on a layout difference rather than a scientific one.
        local = probe_payload(trainer, backend="local", steps=int(steps),
                              shared_checkpoint=Path(shared_checkpoint),
                              device_report={"torch": torch.__version__,
                                             "cuda_available": torch.cuda.is_available()})
    remote_payload = json.loads(Path(remote).read_text(encoding="utf-8"))
    result = compare_backends(local, remote_payload)
    result.update({"experiment_id": experiment_id, "compared_against": against,
                   "scientific_config_hash": parity_row["scientific_config_hash"],
                   "m10_matrix_identity": plan["m10_matrix_identity"]})
    if not dry_run: write_parity(Path(output), result)
    _echo({"passed": result["passed"], "kind": result["kind"],
           "checks": {name: check.get("passed") for name, check in result["checks"].items()},
           "parity_identity": result["parity_identity"],
           "local_device": result["local_backend"]["device"],
           "remote_device": result["remote_backend"]["device"],
           "written": [] if dry_run else [str(output)]})


# --- registry ----------------------------------------------------------------
@app.command("registry")
def registry_command(plan: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--plan",
                                               exists=True, dir_okay=False),
                     root: Path = typer.Option(DEFAULT_REPORTS, "--root"),
                     action: str = typer.Option("init", "--action",
                                                help="init | show | reconcile | lock | validate-lock"),
                     collected: Path = typer.Option(None, "--collected", exists=True, dir_okay=False,
                                                    help="reconcile: the m10_collect_runs payload"),
                     binding: Path = typer.Option(DEFAULT_REPORTS / "B08_M9_REFERENCE_BINDING.json",
                                                  "--binding"),
                     artifacts: str = typer.Option("", "--artifacts",
                                                   help="lock: name=identity pairs, comma separated"),
                     dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Create, inspect, reconcile or freeze the experiment registry."""
    payload = json.loads(Path(plan).read_text(encoding="utf-8"))
    if action == "init":
        registry = ExperimentRegistry.from_plan(Path(root), payload)
        written = registry.write(dry_run=dry_run)
        _echo({"action": "init", **registry.summary(), "written": [] if dry_run else [str(written)]})
        return
    registry = ExperimentRegistry.load(Path(root))
    if action == "show":
        _echo({"action": "show", **registry.summary()}); return
    if action == "reconcile":
        if collected is None: raise typer.BadParameter("--collected is required for reconcile")
        result = _reconcile(registry, json.loads(Path(collected).read_text(encoding="utf-8")))
        written = registry.write(dry_run=dry_run)
        _echo({"action": "reconcile", **result, **registry.summary(),
               "written": [] if dry_run else [str(written)]})
        return
    if action in ("lock", "validate-lock"):
        target = Path(root) / "SOURCE_MATRIX_LOCK.json"
        if action == "lock":
            reference = (json.loads(Path(binding).read_text(encoding="utf-8"))
                         if Path(binding).is_file() else None)
            pairs = dict(item.split("=", 1) for item in artifacts.split(",") if "=" in item)
            lock = source_matrix_lock(registry, payload, code_lineage=_code_lineage(),
                                      reference_binding=reference, artifact_identities=pairs)
            if not dry_run: atomic_json_write(target, lock)
        else:
            lock = json.loads(target.read_text(encoding="utf-8"))
        # Validate twice, as the milestone contract requires: a lock that cannot
        # reproduce its own identity is not a lock.
        first = validate_source_matrix_lock(lock, payload)
        second = validate_source_matrix_lock(lock, payload)
        agreed = first == second
        _echo({"action": action, "passed": first["passed"] and agreed,
               "validated_twice_agree": agreed, "checks": first["checks"],
               "logical_rows": lock["logical_rows"], "rows_by_status": lock["rows_by_status"],
               "failed_rows": lock["failed_rows"],
               "source_matrix_lock_identity": lock["source_matrix_lock_identity"],
               "written": [] if (dry_run or action == "validate-lock") else [str(target)]})
        if not (first["passed"] and agreed): raise typer.Exit(1)
        return
    raise typer.BadParameter("action must be one of init | show | reconcile | lock | validate-lock")


def _code_lineage() -> dict:
    """Which code produced these rows. Read from git, not asserted."""
    import subprocess
    run = lambda *args: subprocess.run(args, capture_output=True, text=True, timeout=15,
                                       check=False).stdout.strip()
    return {"head": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "describe": run("git", "describe", "--always", "--dirty"),
            "recent": run("git", "log", "-8", "--pretty=%h %s")}


def _reconcile(registry, collected: dict) -> dict:
    """Advance the registry from what each run actually WROTE.

    A row is recorded COMPLETED only when its own `run.json` says COMPLETED and it
    names a checkpoint and a calibration; anything else is left alone rather than
    optimistically promoted.
    """
    updated, skipped, failed = [], [], []
    for experiment_id, entry in sorted((collected.get("collected") or {}).items()):
        if experiment_id not in registry.records: skipped.append(experiment_id); continue
        record = registry.get(experiment_id)
        if record.status in ("BLOCKED", "COMPLETED"): skipped.append(experiment_id); continue
        status = entry.get("status")
        if status != "COMPLETED":
            skipped.append(experiment_id); continue
        if not (entry.get("best_checkpoint_sha256") and entry.get("source_calibration_sha256")):
            registry.claim(experiment_id, run_id=experiment_id, backend="modal")
            registry.fail(experiment_id, stage="G6",
                          error="run reported COMPLETED but wrote no checkpoint/calibration")
            failed.append(experiment_id); continue
        if record.status == "PLANNED":
            registry.claim(experiment_id, run_id=experiment_id, backend="modal")
        for stage, marker in (entry.get("stage_outputs") or {}).items():
            registry.set_stage(experiment_id, stage, "COMPLETED")
        registry.update(
            experiment_id, status="COMPLETED", run_id=experiment_id,
            best_checkpoint_path=entry.get("best_checkpoint_path"),
            best_checkpoint_sha256=entry.get("best_checkpoint_sha256"),
            source_calibration_sha256=entry.get("source_calibration_sha256"),
            calibration_hash=entry.get("calibration_hash"),
            source_dev_metrics=entry.get("best_metrics") or {},
            m8_bank_identity=(entry.get("identity") or {}).get("m8_bank_identity"),
            compute={"global_step": entry.get("global_step"), "epoch": entry.get("epoch"),
                     "parameter_counts": entry.get("parameter_counts"),
                     "optimizer_groups": entry.get("optimizer_groups")})
        updated.append(experiment_id)
    return {"reconciled": updated, "reconciled_count": len(updated),
            "recorded_failed": failed, "left_alone": skipped}


# --- G7 ----------------------------------------------------------------------
@app.command("target-predict")
def target_predict(
        package_root: Path = typer.Option(..., "--package-root", exists=True, file_okay=False),
        model_config: Path = typer.Option(Path("configs/models/m9_detector.yaml"), "--model-config",
                                          exists=True, dir_okay=False),
        evaluation_config: Path = typer.Option(DEFAULT_EVALUATION, "--evaluation-config",
                                               exists=True, dir_okay=False),
        loader_config: Path = typer.Option(Path("configs/data/loader_m4.yaml"), "--loader-config",
                                           exists=True, dir_okay=False),
        checkpoint: Path = typer.Option(..., "--checkpoint"),
        weight_root: Path = typer.Option(..., "--weight-root"),
        experiment_id: str = typer.Option(..., "--experiment-id"),
        variant: str = typer.Option("B08", "--variant"),
        seed: int = typer.Option(20260806, "--seed"),
        threshold: float = typer.Option(..., "--threshold", help="the FROZEN source-dev threshold"),
        temperature: float | None = typer.Option(None, "--temperature"),
        calibration_hash: str = typer.Option("", "--calibration-hash"),
        calibration_sha256: str = typer.Option("", "--calibration-sha256"),
        output_root: Path = typer.Option(DEFAULT_REPORTS / "g7", "--output-root"),
        cache_root: Path = typer.Option(Path("data/processed/m10_cache"), "--cache-root"),
        limit: int | None = typer.Option(None, "--limit"),
        batch_size: int = typer.Option(8, "--batch-size"),
        device: str = typer.Option("cpu", "--device"),
        engineering_smoke: bool = typer.Option(False, "--engineering-smoke",
                                               help="label the output as engineering verification, "
                                                    "not a scientific result"),
        dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Predict on target FEATURES. No label is opened, because none is reachable."""
    from prism_fas.data.loader.config import load_loader_config
    from prism_fas.detector.config import detector_config_from, load_yaml
    from prism_fas.detector.heads import resolve_recipe_text_cache
    from prism_fas.detector.pretrained import SigLIP2Artifacts, resolve_convnext_weight
    from prism_fas.detector.prism_detector import build_detector
    firewall = _firewall(evaluation_config)
    evaluation = g7.load_evaluation_config(Path(evaluation_config))
    firewall.assert_cannot_resolve_labels("G7")
    model_payload = load_yaml(Path(model_config))
    detector_config = detector_config_from(model_payload)
    capabilities = g7.VariantCapabilities(has_region=True, has_prompt=True)
    unknown_threshold = evaluation["thresholds"]["unknown_threshold"]
    if dry_run:
        _echo({"status": "dry_run", "experiment_id": experiment_id, "variant": variant,
               "package_root": str(package_root), "limit": limit, "threshold": threshold,
               "unknown_threshold": unknown_threshold,
               "firewall": firewall.report(), "target_labels_opened": False, "written": []})
        return
    siglip = SigLIP2Artifacts.resolve(Path(weight_root))
    text_cache = resolve_recipe_text_cache(Path(weight_root))
    model = build_detector(detector_config, text_embeddings=text_cache.tensor(), siglip=siglip,
                           local_weight_file=resolve_convnext_weight(Path(weight_root)),
                           text_cache_identity=text_cache.identity, device=device)
    opened = g7.load_checkpoint_for_inference(Path(checkpoint), model)
    inference_hash = g7.inference_config_hash(
        variant=variant, flags={"region": "on", "prompt": "frozen_prompt"}, threshold=threshold,
        unknown_threshold=unknown_threshold, temperature=temperature,
        package_identity=evaluation["roots"]["target_feature_package_identity"],
        architecture_identity=model.architecture_identity())
    batches = g7.target_batches(Path(package_root), load_loader_config(Path(loader_config)),
                                cache_root=Path(cache_root), limit=limit, batch_size=batch_size,
                                firewall=firewall,
                                progress=lambda item: typer.echo(json.dumps({"progress": item})))
    rows = g7.predict_target(model, batches, capabilities=capabilities, threshold=threshold,
                             unknown_threshold=unknown_threshold, temperature=temperature,
                             checkpoint_hash=opened["checkpoint_sha256"],
                             calibration_hash=calibration_hash,
                             inference_config_hash=inference_hash, variant=variant, device=device)
    root = Path(output_root) / experiment_id
    predictions_path = g7.write_predictions(root / g7.PREDICTION_FILE, rows, variant=variant,
                                            firewall=None)
    lock = g7.build_prediction_lock(
        experiment_id=experiment_id, variant=variant, seed=seed, rows=rows,
        checkpoint_sha256=opened["checkpoint_sha256"], source_calibration_sha256=calibration_sha256,
        calibration_hash=calibration_hash, inference_config_hash=inference_hash,
        target_feature_package_identity=evaluation["roots"]["target_feature_package_identity"],
        target_package_id=evaluation["roots"]["target_feature_package_id"], threshold=threshold,
        unknown_threshold=unknown_threshold, engineering_smoke=engineering_smoke)
    g7.write_prediction_lock(root / g7.PREDICTION_LOCK_FILE, lock)
    _echo({"status": "completed", "experiment_id": experiment_id, "rows": lock["row_count"],
           "videos": lock["video_count"], "checkpoint": opened,
           "prediction_lock_identity": lock["prediction_lock_identity"],
           "engineering_smoke": lock["engineering_smoke"], "target_labels_opened": False,
           "written": [str(predictions_path), str(root / g7.PREDICTION_LOCK_FILE)]})


@prediction_app.command("validate")
def prediction_validate(predictions: Path = typer.Option(..., "--predictions", exists=True, dir_okay=False),
                        lock: Path | None = typer.Option(None, "--lock")) -> None:
    """Schema, finiteness, applicability and the no-label rule; the lock if given."""
    rows = g7.read_predictions(Path(predictions))
    summary = g7.validate_predictions(rows)
    payload = {"passed": True, **summary,
               "prediction_logical_identity": g7.prediction_logical_identity(rows)}
    if lock is not None:
        stored = json.loads(Path(lock).read_text(encoding="utf-8"))
        payload["lock"] = g7.validate_prediction_lock(stored, rows)
    _echo(payload)


@prediction_app.command("lockset")
def prediction_lockset(root: Path = typer.Option(DEFAULT_REPORTS / "g7", "--root", exists=True,
                                                 file_okay=False),
                       plan: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--plan",
                                                 exists=True, dir_okay=False),
                       registry_root: Path = typer.Option(DEFAULT_REPORTS, "--registry-root"),
                       output: Path = typer.Option(DEFAULT_REPORTS / "TARGET_PREDICTION_LOCKSET.json",
                                                   "--output"),
                       dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Freeze the complete lockset. The label reveal waits behind this file."""
    locks = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted(Path(root).glob(f"*/{g7.PREDICTION_LOCK_FILE}"))]
    if not locks: raise typer.BadParameter(f"no {g7.PREDICTION_LOCK_FILE} under {root}")
    smoke = [lock["experiment_id"] for lock in locks if lock.get("engineering_smoke")]
    if smoke:
        _echo({"passed": False, "reason": "engineering-smoke predictions may not enter the lockset",
               "engineering_smoke": smoke, "written": []})
        raise typer.Exit(1)
    payload = json.loads(Path(plan).read_text(encoding="utf-8"))
    registry = ExperimentRegistry.load(Path(registry_root))
    lockset = g7.build_lockset(locks, matrix_identity=payload["m10_matrix_identity"],
                               registry_identity=registry.identity())
    if not dry_run: atomic_json_write(Path(output), lockset)
    _echo({"passed": True, "entries": lockset["entry_count"],
           "lockset_identity": lockset["lockset_identity"],
           "written": [] if dry_run else [str(output)]})


# --- G8 ----------------------------------------------------------------------
@app.command("score")
def score_command(predictions: Path = typer.Option(..., "--predictions", exists=True, dir_okay=False),
                  lock: Path = typer.Option(..., "--lock", exists=True, dir_okay=False),
                  labels: Path = typer.Option(..., "--labels"),
                  evaluation_config: Path = typer.Option(DEFAULT_EVALUATION, "--evaluation-config",
                                                         exists=True, dir_okay=False),
                  threshold: float = typer.Option(..., "--threshold"),
                  output: Path = typer.Option(DEFAULT_REPORTS / "scoring.json", "--output"),
                  allow_engineering_smoke: bool = typer.Option(False, "--allow-engineering-smoke"),
                  dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """G8. Opens the evaluation-only labels; writes no model state, ever."""
    firewall = _firewall(evaluation_config)
    stored = json.loads(Path(lock).read_text(encoding="utf-8"))
    rows = g7.read_predictions(Path(predictions))
    if dry_run:
        _echo({"status": "dry_run", "predictions": len(rows),
               "lock": g7.validate_prediction_lock(stored, rows),
               "labels_opened": False, "isolation": scoring.isolation_report(), "written": []})
        return
    evaluation = g7.load_evaluation_config(Path(evaluation_config))
    evaluation_labels = scoring.load_evaluation_labels(Path(labels), firewall=firewall, stage="G8")
    result = scoring.score(predictions=rows, lock=stored, labels=evaluation_labels,
                           threshold=threshold,
                           unknown_threshold=evaluation["thresholds"]["unknown_threshold"],
                           allow_engineering_smoke=allow_engineering_smoke)
    written = scoring.write_scoring_report(Path(output), result, firewall=firewall)
    _echo({"status": "completed", "experiment_id": result["experiment_id"],
           "scoring_identity": result["scoring_identity"], "written": [str(written)]})


# --- statistics, reliability, report, acceptance ------------------------------
@app.command("statistics")
def statistics_command(scored: Path = typer.Option(..., "--scored", exists=True, dir_okay=False,
                                                   help="JSON: {experiment_id: {video_ids, scores, labels}}"),
                       plan: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--plan",
                                                 exists=True, dir_okay=False),
                       threshold: float = typer.Option(..., "--threshold"),
                       output: Path = typer.Option(DEFAULT_REPORTS / "statistics.json", "--output"),
                       dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run the five declared comparisons and correct across them."""
    payload = json.loads(Path(plan).read_text(encoding="utf-8"))
    roles = {row["experiment_id"].rsplit("-s", 1)[0]: row["replication_role"] for row in payload["rows"]}
    data = json.loads(Path(scored).read_text(encoding="utf-8"))
    result = bootstrap.run_declared_family(hypotheses=payload["hypotheses"], scored=data, roles=roles,
                                           threshold=threshold)
    if not dry_run: atomic_json_write(Path(output), result)
    _echo({"family": result["declared_family"],
           "computed": [name for name, entry in result["comparisons"].items()
                        if entry.get("status") == "computed"],
           "refused": [name for name, entry in result["comparisons"].items()
                       if entry.get("status") == "refused"],
           "multiple_comparison": result["multiple_comparison"]["policy"],
           "written": [] if dry_run else [str(output)]})


@app.command("reliability")
def reliability_command(output: Path = typer.Option(DEFAULT_REPORTS / "RELIABILITY.json", "--output"),
                        dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Materialize the declared reliability tests and their statuses."""
    payload = reliability.build_report(reliability.declared_tests())
    reliability.write_report(Path(output), payload, dry_run=dry_run)
    _echo({"tests": payload["count"], "by_status": payload["by_status"],
           "blocked": [item["test_id"] for item in payload["blocked"]],
           "uses_target_labels": payload["uses_target_labels"],
           "written": [] if dry_run else [str(output)]})


@app.command("report")
def report_command(plan: Path = typer.Option(DEFAULT_REPORTS / "M10_MATRIX_PLAN.json", "--plan",
                                             exists=True, dir_okay=False),
                   registry_root: Path = typer.Option(DEFAULT_REPORTS, "--registry-root"),
                   reliability_json: Path | None = typer.Option(None, "--reliability"),
                   statistics_json: Path | None = typer.Option(None, "--statistics"),
                   output: Path = typer.Option(DEFAULT_REPORTS / "M10_REPORT.json", "--output"),
                   dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Assemble the report from frozen artifacts. Nothing is recomputed."""
    payload = json.loads(Path(plan).read_text(encoding="utf-8"))
    registry = ExperimentRegistry.load(Path(registry_root))
    assembled = report.assemble(
        plan=payload, registry=registry,
        reliability=(json.loads(Path(reliability_json).read_text(encoding="utf-8"))
                     if reliability_json else None),
        statistics=(json.loads(Path(statistics_json).read_text(encoding="utf-8"))
                    if statistics_json else None),
        target_labels_revealed=False)
    audit = report.audit_no_fabricated_target_values(assembled)
    report.write_report(Path(output), assembled, dry_run=dry_run)
    if not dry_run:
        Path(output).with_suffix(".md").write_text(report.render_markdown(assembled), encoding="utf-8")
    _echo({"report_identity": assembled["report_identity"], "sections": len(assembled["sections"]),
           "no_fabricated_target_values": audit, "target_labels_revealed": False,
           "written": [] if dry_run else [str(output), str(Path(output).with_suffix(".md"))]})


@app.command("acceptance")
def acceptance_command(config: Path = typer.Option(DEFAULT_MATRIX, "--config", exists=True, dir_okay=False),
                       evaluation_config: Path = typer.Option(DEFAULT_EVALUATION, "--evaluation-config",
                                                              exists=True, dir_okay=False),
                       root: Path = typer.Option(DEFAULT_REPORTS, "--root"),
                       output: Path = typer.Option(DEFAULT_REPORTS / "M10_ACCEPTANCE.json", "--output"),
                       dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """What is and is not yet true about M10, checked rather than asserted."""
    firewall = _firewall(evaluation_config)
    first = experiment_matrix.build_plan(Path(config))
    second = experiment_matrix.build_plan(Path(config))
    agreement = experiment_matrix.plans_agree(first, second)
    checks = {
        "matrix_deterministic": agreement["identical"],
        "blocked_rows_carry_reasons": all(row.get("blocked_reason") for row in first["rows"]
                                          if row["status"] == "BLOCKED"),
        "unique_experiment_ids": len({row["experiment_id"] for row in first["rows"]}) == len(first["rows"]),
        "train_cannot_read_target_labels": firewall.permission("TRAIN", "target_label_root") == "deny",
        "g7_cannot_read_target_labels": firewall.permission("G7", "target_label_root") == "deny",
        "g8_reads_target_labels": firewall.permission("G8", "target_label_root") == "read",
        "g8_has_no_training_capability": scoring.isolation_report()["static_import_audit"]["passed"],
        "target_labels_revealed_is_false": not Path(root, "TARGET_LABEL_REVEAL.json").exists()}
    payload = {"m10_acceptance_schema_version": "m10-acceptance-v1",
               "m10_matrix_identity": first["m10_matrix_identity"], "checks": checks,
               "passed": all(checks.values()), "summary": first["summary"],
               "target_labels_revealed": False,
               "not_claimed": ["SiW-Mv2 performance", "cross-domain superiority",
                               "ablation superiority", "state-of-the-art comparison"]}
    if not dry_run: atomic_json_write(Path(output), payload)
    _echo({**{key: payload[key] for key in ("passed", "checks", "m10_matrix_identity")},
           "written": [] if dry_run else [str(output)]})
    if not payload["passed"]: raise typer.Exit(1)
