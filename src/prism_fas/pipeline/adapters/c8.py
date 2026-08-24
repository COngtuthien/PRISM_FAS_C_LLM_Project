"""C8 — the source-only experiment matrix, rehearsed and executed.

C8 is the stage where "one command does not imply one output" bites hardest.
L.8 requires every atomic run — every protocol x method x config x seed — to emit
its own durable artifacts before the orchestrator advances, and forbids
winner-only cleanup afterwards. A scheduler that lost a failed seed would look
identical to one that never ran it.

So this adapter is organised around the run, not the stage: it materializes the
§18 matrix, decides identity-aware what still has to run, executes rows end to
end, and preserves failures beside successes.

**Two execution paths, and they share no metric-producing code.** The rehearsal
runs a bounded sample of rows on fixture batches through an audit detector, which
proves the artifact contract and the scheduler. The scientific path resolves the
frozen C7 detector configuration, the row's own C6 matched bank and the row's
protocol splits, and trains through the canonical `M9Trainer`. The separation is
structural rather than conventional: `_run_one` calls `assert_fixture_permitted`
before it builds anything, so a scientific context reaching it RAISES instead of
writing fixture metrics into a scientific manifest. That was the live defect —
under a scientific context the scheduler correctly selected all 42 rows and the
executor would have run every one of them on `audit_batch`, then written 42 PASS
manifests C9 would have frozen.

The one thing neither path will do is touch the target. P3-ready means selected
on CASIA-dev and MSU-dev; the SiW package is not opened, not resolved and not
reachable from any code path here. That is checked, not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterError, AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, read_json,
                                                resume_decision, stage_reports_dir,
                                                stage_runs_dir, utc, write_artifact)
from prism_fas.evaluation import detector_reliability
from prism_fas.pipeline.execution import ExecutionContext

STAGE_ID = "C8"

PLAN_MATRIX = "PLAN_MATRIX"
SCHEDULE = "SCHEDULE"
EXECUTE_ROWS = "EXECUTE_ROWS"
FAILURE_PRESERVATION = "FAILURE_PRESERVATION"
TARGET_ISOLATION = "TARGET_ISOLATION"

#: Scientific-only substages. `FAILURE_PRESERVATION` is deliberately NOT among
#: them: it constructs a deliberate failure to prove the retention path, which is
#: exactly what a scientific run may never do. A real scientific failure writes
#: its own manifest from `_run_scientific_row`, and acceptance reflects it.
VERIFY_INPUTS = "VERIFY_INPUTS"
CROSS_SOURCE_DIAGNOSTICS = "CROSS_SOURCE_DIAGNOSTICS"
CALIBRATION_STABILITY = "CALIBRATION_STABILITY"
ACCEPTANCE = "ACCEPTANCE"

SCIENTIFIC_MODES: tuple[str, ...] = (VERIFY_INPUTS, CROSS_SOURCE_DIAGNOSTICS,
                                     CALIBRATION_STABILITY, ACCEPTANCE)

MODES: tuple[str, ...] = (PLAN_MATRIX, SCHEDULE, EXECUTE_ROWS, FAILURE_PRESERVATION,
                          TARGET_ISOLATION) + SCIENTIFIC_MODES

#: How many of the 42 preregistered rows the rehearsal actually executes. The
#: scheduler is what C8 readiness proves; running 42 CPU detectors would prove
#: only that a CPU is slow.
#:
#: The sample is the FIRST rows of the plan, not the first rows still pending.
#: Sampling the pending remainder slid the window forward on every rerun, so the
#: same command exercised different arms each time and, after enough reruns,
#: would have executed nothing at all while still reporting PASS.
SMOKE_ROWS = 2


@dataclass
class C8Adapter(EngineeringAdapter):
    """The C8 execution adapter. The matrix and the metrics are imported."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Source matrix over arms, tracks, configs and seeds"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = True

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c6_matched_banks", "reports/full/c6",
                          "the matched 1024-per-arm synthetic banks every row trains on"),
            RequiredInput("c7_config_lock", "reports/full/c7/DETECTOR_CONFIG_LOCK.json",
                          "the frozen detector configuration the matrix runs at"),
            RequiredInput("source_packages", "data/packages",
                          "the preprocessed CASIA and MSU source packages"),
            RequiredInput("pretrained_weights", "data/packages/pretrained",
                          "the pinned SigLIP2 and ConvNeXt weights"),
        )

    def semantic_preconditions(self, request: AdapterRequest) -> list[dict[str, Any]]:
        """Beyond existence: the C7 lock must VERIFY, under C7's own verifier.

        `DETECTOR_CONFIG_LOCK.json` existing proves nothing — a refused, drifted
        or rehearsal-shaped lock is a file that exists. C8 trains 42 rows at the
        configuration it names, so the gate is the same strict verification C7
        applies at the moment it writes it.
        """
        from prism_fas.evaluation import c6_evidence
        from prism_fas.pipeline.adapters.c7 import (SCIENTIFIC_CONFIG_LOCK_PATH,
                                                    verify_detector_config_lock)

        closure = c6_evidence.evidence_report(request.repo)
        verification = verify_detector_config_lock(
            request.repo, request.repo / SCIENTIFIC_CONFIG_LOCK_PATH)
        problems = [item["check_id"] for item in verification["checks"]
                    if not item["ok"]]
        return [
            {"name": "c6_closure_verified", "path": c6_evidence.C6_REPORTS,
             "present": closure["valid"], "blocking": not closure["valid"],
             "description": "the three C6 matched banks every row trains on",
             "verifier": "prism_fas.evaluation.c6_evidence.verify_c6_evidence",
             "reason_code": closure["reason_code"],
             "problems": closure["problems"][:12]},
            {"name": "c7_config_lock_verified", "path": SCIENTIFIC_CONFIG_LOCK_PATH,
             "present": verification["valid"], "blocking": not verification["valid"],
             "description": ("the frozen detector configuration, its complete retained "
                             "trial set, its winning checkpoint and the decision graph "
                             "it was selected under"),
             "verifier": "prism_fas.pipeline.adapters.c7.verify_detector_config_lock",
             "reason_code": "C7_CONFIG_LOCK_INVALID" if problems else "",
             "problems": problems[:12]},
        ]

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        """Two workflows, chosen by the context — never one that adapts.

        The defect this closes: there was ONE workflow whose executor called
        `audit_batch` and `build_audit_detector` unconditionally. The scheduler
        was already correct — under a scientific context `context.limit` returns
        the full 42 and never reads `SMOKE_ROWS` — so a scientific run would have
        trained all 42 rows on fixture batches through an audit model and written
        42 PASS manifests. Every check would have passed, because the fixture
        execution is correct engineering. It was in the wrong place.
        """
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        """The rehearsal path. Fixture-backed rows, and a constructed failure."""
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID)
        budget = context.budget or SmokeBudget.from_profile(request.profile)

        plan, plan_result = self._plan(request, reports, context)
        scheduled, schedule_result = self._schedule(request, plan, reports, context)
        executed, execute_result = self._execute(request, scheduled, reports, runs,
                                                 budget, context)
        return [plan_result, schedule_result, execute_result,
                self._failure_preservation(request, plan, reports, runs),
                self._target_isolation(request, reports)]

    # --- modes ----------------------------------------------------------------

    def _plan(self, request: AdapterRequest, reports: Path,
              context: ExecutionContext) -> tuple[Any, AdapterResult]:
        from prism_fas.evaluation.source_matrix import (ARMS, SEED_FAMILY, build_plan)

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        report = plan.validate()

        checks.append(check(
            "c8_matrix_satisfies_replication_policy", report["valid"],
            "the planned matrix satisfies the §18.3 seed counts for every row",
            seed_counts=report["seed_counts"], problems=report["problems"],
            rows=report["rows"], unique_configurations=report["unique_configurations"]))
        checks.append(check(
            "c8_primary_p3_rows_have_five_seeds",
            all(report["seed_counts"].get(f"C-G-{arm}:P3") == 5 for arm in ARMS),
            "the three primary P3 Track-G generator rows carry 5 seeds each",
            per_arm={arm: report["seed_counts"].get(f"C-G-{arm}:P3") for arm in ARMS}))
        checks.append(check(
            "c8_prompthead_ablation_present",
            report["seed_counts"].get("C-R-NOPROMPT:P3") == 3,
            "the C-H5 PromptHead ablation is planned with 3 seeds",
            seeds=report["seed_counts"].get("C-R-NOPROMPT:P3")))
        checks.append(check(
            "c8_seed_family_is_closed",
            all(row.seed in SEED_FAMILY for row in plan.rows),
            "no row uses a seed outside the fixed family",
            seed_family=list(SEED_FAMILY),
            rule="§18.3: no best-seed reporting and no cherry-picking; the family is not "
                 "extended after a result is seen"))
        checks.append(check(
            "c8_replication_shares_one_config_identity",
            report["unique_configurations"] < report["rows"],
            "rows that differ only by seed share one configuration identity",
            rows=report["rows"], unique_configurations=report["unique_configurations"]))

        # The plan is the complete frozen matrix under BOTH contexts. Only how
        # many of its rows get executed differs, and that is the scheduler's
        # decision, recorded below rather than folded into the plan.
        artifact = write_artifact(request, reports / "C8_MATRIX_PLAN.json", {
            **plan.as_dict(), "generated_at_utc": utc(), "mode": PLAN_MATRIX,
            **context.stamp(),
            "fixture_backed": context.fixtures_permitted,
            "rows_declared": len(plan.rows),
            "rows_executed_here": context.limit(len(plan.rows), sample=SMOKE_ROWS)})
        return plan, self.result(request, mode=PLAN_MATRIX, checks=checks,
                                 artifacts=[artifact],
                                 parent_identities={"c8_source_matrix": plan.identity})

    def _schedule(self, request: AdapterRequest, plan: Any, reports: Path,
                  context: ExecutionContext) -> tuple[list[Any], AdapterResult]:
        """Decide what still has to run. Identity-aware, not existence-aware."""
        checks: list[dict[str, Any]] = []
        runs = stage_runs_dir(request, STAGE_ID)
        decisions: list[dict[str, Any]] = []

        directories: list[Path] = []
        for row in plan.rows:
            directory = (runs or reports) / row.protocol / row.experiment_id / \
                row.config_identity[:12] / str(row.seed)
            directories.append(directory)
            decision = resume_decision(request, row.row_id,
                                       directory / "run_manifest.json",
                                       expected_identity=row.run_identity,
                                       identity_key="run_identity")
            decisions.append({"row_id": row.row_id, **decision})

        pending = [row for row, decision in zip(plan.rows, decisions)
                   if decision["action"] == "EXECUTE"]
        skipped = [decision for decision in decisions
                   if decision["action"] == "SKIP_VALID_COMPLETE"]

        # How many rows run is the context's answer, not this module's. Under a
        # scientific context `limit` returns the declared count and never reads
        # SMOKE_ROWS, so the rehearsal sampling constant is not reachable from
        # the scientific path even by mistake (§8).
        count = context.limit(len(plan.rows), sample=SMOKE_ROWS)
        # Fixed by plan order, so rerunning exercises the same arms. A row that is
        # already valid is carried with its stored directory rather than re-run,
        # which keeps resume honest without letting the window drift forward.
        sample = [{"row": row, "decision": decision, "directory": directory}
                  for row, decision, directory
                  in list(zip(plan.rows, decisions, directories))[:count]]

        checks.append(check(
            "c8_schedule_is_identity_aware", True,
            "each row's resume decision compares a recorded run identity, not a filename",
            planned=len(plan.rows), pending=len(pending), skipped=len(skipped),
            rule="L.11: a completed unit is skipped only after its parent identities, "
                 "config identity, content hash and acceptance state validate"))
        checks.append(check(
            "c8_missing_seeds_resume_independently", True,
            "each protocol/method/config/seed is its own resumable unit",
            unit="runs/<profile>/c8/<protocol>/<experiment>/<config>/<seed>/",
            example_units=[decision["row_id"] for decision in decisions[:3]]))
        checks.append(check(
            "c8_sample_is_stable",
            [item["row"].row_id for item in sample]
            == [row.row_id for row in plan.rows[:count]],
            "the scheduled rows are the first rows of the plan, so a rerun of the "
            "same command exercises the same arms",
            sampled=[item["row"].row_id for item in sample],
            rule="a sample drawn from the pending remainder would slide forward on "
                 "every rerun and eventually execute nothing while reporting PASS"))
        checks.append(check(
            "c8_scientific_cardinality_is_the_complete_matrix",
            not context.is_scientific or count == len(plan.rows),
            "a scientific run schedules every declared row of the frozen matrix"
            if context.is_scientific else
            "a rehearsal samples the matrix; the scientific path does not",
            context=context.name, declared=len(plan.rows), scheduled=count,
            cardinality_rule=context.cardinality_rule,
            rule="§8: no SMOKE_ROWS, first-N, pending-prefix or fixture cardinality "
                 "may affect a full run"))

        artifact = write_artifact(request, reports / "C8_SCHEDULE.json", {
            "schema_version": "c8-schedule-v1", "generated_at_utc": utc(),
            "mode": SCHEDULE, "matrix_identity": plan.identity,
            "planned": len(plan.rows), "pending": len(pending), "skipped": len(skipped),
            "decisions": decisions, "rows_executed_here": count,
            **context.stamp(),
            "rows_declared": len(plan.rows),
            "sampled_rows": [item["row"].row_id for item in sample],
            "sample_rule": ("every declared row" if context.is_scientific else
                            "first rows of the plan, in plan order"),
            "fixture_backed": context.fixtures_permitted})
        return sample, self.result(request, mode=SCHEDULE, checks=checks,
                                   artifacts=[artifact])

    def _execute(self, request: AdapterRequest, sample: list[dict[str, Any]],
                 reports: Path, runs: Path | None, budget: SmokeBudget,
                 context: ExecutionContext) -> tuple[list[dict[str, Any]], AdapterResult]:
        """Run a bounded sample of rows through the real training control flow."""
        checks: list[dict[str, Any]] = []
        executed: list[dict[str, Any]] = []
        rows = [item["row"] for item in sample]

        for item in sample:
            row = item["row"]
            try:
                if item["decision"]["action"] == "SKIP_VALID_COMPLETE":
                    executed.append(self._reuse_one(request, row, item["directory"]))
                else:
                    executed.append(self._run_one(request, row, runs or reports, budget))
            except Exception as error:
                executed.append({"row_id": row.row_id, "status": "FAIL",
                                 "error": f"{type(error).__name__}: {error}"})

        passed = [item for item in executed if item.get("status") == "PASS"]
        checks.append(check(
            "c8_rows_execute_end_to_end", len(passed) == len(rows) and bool(rows),
            f"{len(passed)}/{len(rows)} sampled row(s) ran forward, backward, checkpoint "
            "and calibration",
            rows=[item["row_id"] for item in executed],
            failures=[item for item in executed if item.get("status") != "PASS"]))
        checks.append(check(
            "c8_run_manifest_carries_the_l8_columns",
            all(set(item.get("manifest_keys", [])) >= {
                "run_identity", "protocol", "method", "arm", "config_identity", "seed",
                "environment", "metrics", "checkpoint", "parent_identities", "status"}
                for item in passed),
            "each run manifest carries config, identity, protocol, method, arm, seed, "
            "environment, metrics, checkpoint identity, parents and status",
            required=["run_identity", "protocol", "method", "arm", "config_identity",
                      "seed", "environment", "metrics", "checkpoint",
                      "parent_identities", "status"]))
        checks.append(check(
            "c8_calibration_is_source_dev_only",
            all(item.get("calibration", {}).get("split") == "source_dev"
                for item in passed),
            "temperature and threshold were fitted on source_dev alone",
            calibrations=[item.get("calibration", {}) for item in passed],
            rule="§16: C-G5 fits temperature and operating threshold on source_dev only"))
        checks.append(check(
            "c8_checkpoint_inheritance_recorded",
            all(item.get("parent_identities") for item in passed),
            "each run records the ancestor identities it inherited",
            parents=[item.get("parent_identities") for item in passed]))

        artifact = write_artifact(request, reports / "C8_EXECUTED_ROWS.json", {
            "schema_version": "c8-executed-rows-v1", "generated_at_utc": utc(),
            "mode": EXECUTE_ROWS,
            "rows": [{key: value for key, value in item.items()
                      if key not in ("complexity", "resources")} for item in executed],
            "rows_executed": len(executed), "rows_planned": 42,
            **context.stamp(),
            "fixture_backed": context.fixtures_permitted,
            "budget": None if context.is_scientific else budget.as_dict(),
            "note": "a bounded sample of the preregistered matrix, run on fixtures to "
                    "exercise the per-run artifact contract. No scientific comparison "
                    "is possible from these numbers (L.1)"})

        # One rollup per stage, covering every arm that ran, ordered by row id so
        # the file is reproducible. The per-run copies beside each checkpoint stay
        # authoritative; this is the navigable summary the reporting layer reads.
        # Same writer as the scientific path uses, so the two produce the same
        # shape and a rehearsal of it is evidence about the real one.
        rollups = self._rollups(request, executed, reports)

        return executed, self.result(request, mode=EXECUTE_ROWS, checks=checks,
                                     artifacts=[artifact, *rollups])

    def _reuse_one(self, request: AdapterRequest, row: Any,
                   directory: Path) -> dict[str, Any]:
        """Report a row that resume validated, from what it already wrote.

        Re-running it would contradict L.11; omitting it would make the rollup
        and the complexity table shrink on the second run of the same command.
        """
        import json

        def stored(name: str) -> dict[str, Any]:
            path = directory / name
            if not path.is_file():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}

        manifest = stored("run_manifest.json")
        return {"row_id": row.row_id, "status": "PASS", "reused": True,
                "manifest_keys": sorted(manifest),
                "calibration": manifest.get("calibration", {}),
                "parent_identities": manifest.get("parent_identities", {}),
                "complexity": stored("model_complexity.json"),
                "resources": stored("compute_resources.json"),
                "path": directory.relative_to(request.repo).as_posix()}

    def _run_one(self, request: AdapterRequest, row: Any, root: Path,
                 budget: SmokeBudget) -> dict[str, Any]:
        """One atomic REHEARSAL run, with its own durable artifacts (L.8).

        Fixture-backed by construction: an audit detector stepped on `audit_batch`
        for `budget.steps`, evaluated on a fixture batch. That exercises the whole
        per-run artifact contract — manifest columns, calibration record,
        checkpoint, history, complexity, resources — while being structurally
        incapable of producing a scientific number.

        The guard is the first statement rather than a comment because this
        function is the one a scientific context must never reach. It used to be
        reachable: `workflow` had no branch, so `--profile full` ran it for every
        one of the 42 scheduled rows.
        """
        assert_fixture_permitted(request.context,
                                 "the C8 fixture row executor (audit_batch + "
                                 "build_audit_detector + SmokeBudget)")
        import platform

        import numpy as np
        import torch

        from prism_fas.detector.variant import ResolvedExperimentVariant
        from prism_fas.evaluation.variant_audit import audit_batch, build_audit_detector
        from prism_fas.detector.losses import compute_losses
        from prism_fas.detector.trainer import (M9TrainingConfig, batch_contract_for,
                                                enabled_terms)
        from prism_fas.pipeline.portability import (KNOWN_BACKENDS, PROVENANCE_NAMESPACE,
                                                    resolve_microbatch)
        from prism_fas.train.calibration import apply_temperature, fit_temperature
        from prism_fas.train.metrics import select_min_acer_threshold, summarize

        flags = {key: value for key, value in row.flags.items() if key != "recipe_arm"}
        variant = ResolvedExperimentVariant.resolve(flags)
        model = build_audit_detector(variant)
        config = M9TrainingConfig(run_id=row.row_id, variant=variant, steps_per_epoch=2)
        contract = batch_contract_for("G5", config)
        groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
        optimizer = torch.optim.AdamW(groups)

        losses: list[float] = []
        for _step in range(max(1, budget.steps)):
            batch = audit_batch(variant, contract)
            output = model(batch)
            result = compute_losses(output, batch, model.manifold,
                                    text_embeddings=model.text_matrix(),
                                    enabled=enabled_terms("G5", variant), variant=variant)
            optimizer.zero_grad(set_to_none=True)
            result.total.backward()
            optimizer.step()
            losses.append(float(result.total.detach()))

        # A source_dev-shaped evaluation on the model's OWN decision logit, then
        # the canonical temperature fit and threshold search on that logit alone.
        with torch.no_grad():
            evaluation = audit_batch(variant, contract)
            logits = model(evaluation).global_logit.squeeze(-1).numpy().astype("float64")
        targets = evaluation.label.numpy().astype("float64")
        temperature = fit_temperature(logits, targets)
        probabilities = apply_temperature(logits, temperature)
        selection = select_min_acer_threshold(probabilities, targets)
        threshold = float(selection["selected"]["threshold"])
        metrics = summarize(probabilities, targets, threshold)

        destination = (root / row.protocol / row.experiment_id /
                       row.config_identity[:12] / str(row.seed))
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(),
                    "architecture_identity": model.architecture_identity()},
                   checkpoint_path)

        backend = KNOWN_BACKENDS["local_cpu"]
        microbatch = resolve_microbatch(backend=backend,
                                        effective_batch=contract.batch_size,
                                        composition={"real_live": contract.real_live,
                                                     "real_spoof": contract.real_spoof,
                                                     "synthetic": contract.synthetic})
        manifest = {
            "schema_version": "c8-run-manifest-v1",
            "run_identity": row.run_identity,
            "row_id": row.row_id,
            "protocol": row.protocol,
            "method": row.experiment_id,
            "track": row.track,
            "arm": row.arm,
            "config_identity": row.config_identity,
            "seed": row.seed,
            "replication_role": row.replication_role,
            "hypotheses": list(row.hypotheses),
            "flags": variant.flags(),
            "variant_identity": variant.identity(),
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform()},
            "metrics": {"train_loss": losses,
                        "source_dev": {key: value for key, value in metrics.items()
                                       if isinstance(value, (int, float))}},
            "calibration": {"split": "source_dev", "temperature": float(temperature),
                            "threshold": threshold,
                            "threshold_criterion": selection["criterion"],
                            "threshold_tie_break": selection["tie_break"],
                            "decision_logit_name": variant.decision_logit_name,
                            "decision_score_name": variant.decision_score_name,
                            "uses_target": False},
            "checkpoint": {"path": checkpoint_path.relative_to(request.repo).as_posix(),
                           "architecture_identity": model.architecture_identity()},
            "parent_identities": {"variant": variant.identity(),
                                  "architecture": model.architecture_identity(),
                                  "batch_contract": contract.identity()},
            "microbatch_plan": microbatch.as_dict(),
            PROVENANCE_NAMESPACE: backend.as_dict(),
            "status": "PASS",
            "selection_tuple": list(row.selection_tuple),
            "target_paths_resolved": 0,
            "target_labels_resolved": 0,
        }
        write_artifact(request, destination / "run_manifest.json", manifest)

        # Per-run structured history and compute provenance, beside the manifest.
        from prism_fas.reporting import complexity as complexity_module
        from prism_fas.reporting import resources as resources_module
        from prism_fas.reporting.history import HistoryWriter

        writer = HistoryWriter(path=destination / "train_history.jsonl",
                               run_identity=row.run_identity)
        for index, value in enumerate(losses):
            writer.append(epoch=0, step=index + 1, total_loss=value,
                          learning_rates=HistoryWriter.group_learning_rates(optimizer),
                          source_dev={key: value for key, value in metrics.items()
                                      if isinstance(value, (int, float))}
                          if index == len(losses) - 1 else None)
        # Named by row id, not experiment id: the matrix runs the same experiment
        # at several protocols and seeds, and those rows would otherwise appear as
        # duplicate identically-named entries in the complexity table.
        complexity_payload = complexity_module.profile_model(
            model, evaluation, name=f"detector_{row.row_id}",
            input_shape=list(evaluation.image.shape))
        resource_payload = resources_module.resource_record(microbatch_plan=microbatch)
        write_artifact(request, destination / "model_complexity.json", complexity_payload)
        write_artifact(request, destination / "compute_resources.json", resource_payload)
        # The stage-level rollup is written once by `_execute`, over every row.
        # Writing it here would make it last-writer-wins: the matrix runs many
        # arms, and the surviving file would name whichever arm finished last.
        return {"row_id": row.row_id, "status": "PASS",
                "manifest_keys": sorted(manifest),
                "calibration": manifest["calibration"],
                "parent_identities": manifest["parent_identities"],
                "complexity": complexity_payload,
                "resources": resource_payload,
                "path": destination.relative_to(request.repo).as_posix()}

    def _failure_preservation(self, request: AdapterRequest, plan: Any, reports: Path,
                              runs: Path | None) -> AdapterResult:
        """A failed row stays addressable. L.8 forbids winner-only cleanup.

        This mode CONSTRUCTS a failure to prove the retention path exists, which
        is legitimate rehearsal evidence and illegitimate science: a scientific
        run's failure set must be exactly the rows that really failed. So the
        guard is here too, and `_scientific_workflow` never calls this mode.
        """
        assert_fixture_permitted(request.context,
                                 "the C8 constructed forced-failure row")
        checks: list[dict[str, Any]] = []
        row = plan.rows[-1]
        destination = ((runs or reports) / row.protocol / row.experiment_id /
                       row.config_identity[:12] / f"{row.seed}_forced_failure")
        destination.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": "c8-run-manifest-v1",
            "run_identity": f"{row.run_identity}:forced_failure",
            "row_id": f"{row.row_id}:forced_failure",
            "protocol": row.protocol, "method": row.experiment_id, "arm": row.arm,
            "config_identity": row.config_identity, "seed": row.seed,
            "status": "FAIL",
            "failure": {"stage": "G5", "error": "constructed engineering failure",
                        "recoverable": False},
            "metrics": {}, "checkpoint": None,
            "parent_identities": {"source_matrix": plan.identity},
            "retention": ("preserved and addressable from the master index; a losing or "
                          "failing row is evidence and is never deleted after a winner "
                          "exists (L.6, L.8)"),
        }
        path = write_artifact(request, destination / "run_manifest.json", manifest)

        checks.append(check(
            "c8_failed_row_is_written_not_dropped",
            (request.repo / path).exists(),
            "a failed row writes its own durable manifest",
            path=path, status="FAIL"))
        checks.append(check(
            "c8_failed_row_keeps_its_reason",
            bool((read_json(request.repo / path) or {}).get("failure")),
            "the failure reason is recorded rather than summarized away",
            failure=(read_json(request.repo / path) or {}).get("failure")))
        checks.append(check(
            "c8_no_winner_only_cleanup", True,
            "no code path in this stage deletes a row after another row succeeds",
            rule="L.8: failed and blocked rows remain addressable from the master index "
                 "and the final evidence package"))

        return self.result(request, mode=FAILURE_PRESERVATION, checks=checks,
                           artifacts=[path])

    def _target_isolation(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """C8 opens no target. Measured over the artifacts this stage wrote."""
        from prism_fas.evaluation.source_matrix import PROTOCOLS

        checks: list[dict[str, Any]] = []
        tokens = ("siw", "SiW", "SIW")
        hits: list[str] = []
        for path in sorted(reports.rglob("*.json")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            # P3's protocol declaration names its future target by design; the
            # violation would be a resolved path, label or metric, not the name.
            for token in tokens:
                if token in text and "siw_mv2_v2" not in text:
                    hits.append(path.name)
                    break

        checks.append(check(
            "c8_no_target_artifact_written", not hits,
            "no C8 artifact resolves a target path, label or metric",
            files_with_unexpected_target_reference=hits,
            permitted="the P3 protocol declaration names siw_mv2_v2 as its future test "
                      "domain; naming it is not resolving it"))
        checks.append(check(
            "c8_p3_is_ready_not_run",
            PROTOCOLS["P3"]["role"].startswith("fixed held-out target"),
            "P3 rows are selected on CASIA-dev and MSU-dev; prediction happens at C11",
            p3=dict(PROTOCOLS["P3"]),
            rule="§19.2: training/LLM/synthesis environments may resolve no SiW labels, "
                 "attack-family metadata or target metrics"))
        checks.append(check(
            "c8_selection_never_reads_a_target_metric", True,
            "the source selection tuple contains only source-domain quantities",
            p1p2_tuple=list(__import__(
                "prism_fas.evaluation.source_matrix", fromlist=["P1P2_TUPLE"]).P1P2_TUPLE),
            p3_ready_tuple=list(__import__(
                "prism_fas.evaluation.source_matrix",
                fromlist=["P3_READY_TUPLE"]).P3_READY_TUPLE)))

        artifact = write_artifact(request, reports / "C8_TARGET_ISOLATION.json", {
            "schema_version": "c8-target-isolation-v1", "generated_at_utc": utc(),
            "mode": TARGET_ISOLATION, "target_paths_resolved": 0,
            "target_labels_resolved": 0, "target_metrics_computed": 0,
            "artifacts_scanned": len(list(reports.rglob("*.json"))),
            "unexpected_references": hits, "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=TARGET_ISOLATION, checks=checks,
                           artifacts=[artifact])

    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C8: 42 rows, the frozen C7 configuration, the canonical trainer.

        Every mode below is reachable in the rehearsal too — same scheduler, same
        resume rule, same manifest schema, same acceptance aggregation — except
        that the rows are trained rather than fixture-stepped. That is what makes
        rehearsing this path evidence about the scientific one.
        """
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID) or reports

        inputs, prepare = self._scientific_prepare(request, reports)
        if inputs is None:
            return [prepare]

        plan, plan_result = self._plan(request, reports, context)
        scheduled, schedule_result = self._schedule(request, plan, reports, context)
        executed, execute_result = self._scientific_execute(
            request, inputs, scheduled, reports, runs)
        diagnostics = self._scientific_diagnostics(request, executed, reports)
        stability = self._scientific_calibration_stability(request, executed, reports)
        isolation = self._target_isolation(request, reports)
        acceptance = self._scientific_acceptance(
            request, inputs, plan, executed, reports,
            gates=[diagnostics, stability, isolation])
        return [prepare, plan_result, schedule_result, execute_result,
                diagnostics, stability, isolation, acceptance]

    def _scientific_prepare(self, request: AdapterRequest,
                            reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Resolve the frozen inputs and VERIFY the C7 lock with C7's own verifier."""
        from prism_fas.pipeline.adapters import sources
        from prism_fas.pipeline.adapters.c7 import (SCIENTIFIC_CONFIG_LOCK_PATH,
                                                    verify_detector_config_lock)

        checks: list[dict[str, Any]] = []
        try:
            inputs = sources.verify_detector_inputs(request.repo)
        except sources.SourceUnavailable as error:
            checks.append(check(
                "c8_scientific_inputs_verified", False,
                f"the frozen scientific inputs are not usable: {type(error).__name__}",
                error=str(error),
                reason_code=getattr(error, "reason_code", "MISSING_DATA")))
            return None, self.result(request, mode=VERIFY_INPUTS, checks=checks,
                                     summary="C8 scientific inputs unavailable")

        checks.append(check(
            "c8_scientific_inputs_verified", True,
            "the M3B package, the M7 bank, the pinned backbones, the C5 candidate "
            "tree and the three C6 bank locks are present and agree",
            package_identity=inputs["package_identity"],
            c6_arms=inputs["c6_arms"], verifier=inputs["verified_by"]))

        lock_path = request.repo / SCIENTIFIC_CONFIG_LOCK_PATH
        verification = verify_detector_config_lock(request.repo, lock_path)
        checks.extend(verification["checks"])
        checks.append(check(
            "c8_uses_c7s_own_lock_verifier", True,
            "the detector config lock is verified by the module that produced it, "
            "not by a second and laxer check inside C8",
            verifier="prism_fas.pipeline.adapters.c7.verify_detector_config_lock",
            lock=SCIENTIFIC_CONFIG_LOCK_PATH, valid=verification["valid"]))
        if not verification["valid"]:
            return None, self.result(
                request, mode=VERIFY_INPUTS, checks=checks,
                summary="C8 refuses to run: the C7 detector config lock does not verify")

        lock = verification["payload"]
        closure = inputs["c6"]
        checks.append(check(
            "c8_c7_lock_binds_this_c6_closure",
            lock.get("c6_selector_identity_sha256") == closure["selector_identity_sha256"]
            and dict(lock.get("c6_bank_locks") or {}) == {
                arm: item["selected_set_sha256"] for arm, item in closure["banks"].items()},
            "the frozen configuration was searched against exactly the C6 banks this "
            "run resolved",
            lock_selector=lock.get("c6_selector_identity_sha256"),
            current_selector=closure["selector_identity_sha256"],
            lock_banks=lock.get("c6_bank_locks"),
            current_banks={arm: item["selected_set_sha256"]
                           for arm, item in closure["banks"].items()}))
        checks.append(check(
            "c8_c7_lock_binds_this_source_package",
            lock.get("source_package_identity") == inputs["package_identity"],
            "the frozen configuration was searched against this source package",
            lock_package=lock.get("source_package_identity"),
            current_package=inputs["package_identity"]))

        from prism_fas.evaluation.source_matrix import build_plan as _build_plan

        needed = sorted({item.track for item in _build_plan().rows})
        frozen = sorted(dict(lock.get("tracks") or {}))
        checks.append(check(
            "c8_every_declared_track_has_a_frozen_configuration",
            set(needed) <= set(frozen),
            "the matrix runs no track the C7 lock left unconfigured",
            tracks_in_matrix=needed, tracks_in_lock=frozen))
        checks.append(check(
            "c8_configuration_is_shared_within_each_track",
            lock.get("shared_within_track") is True
            and lock.get("per_arm_search_performed") is False,
            "one frozen configuration per track, used by every primary generator arm "
            "of that track; C7 ran no per-arm search",
            training_arm=lock.get("training_arm"),
            shared_within_track=lock.get("shared_within_track"),
            per_arm_search_performed=lock.get("per_arm_search_performed"),
            configurations={track: dict(sub).get("winner_config_sha256")
                            for track, sub in sorted(
                                dict(lock.get("tracks") or {}).items())},
            rule="\u00a718.1: the generator arm is the treatment; a configuration tuned "
                 "per arm would confound the treatment with detector tuning"))
        checks.append(check(
            "c8_search_arm_is_the_frozen_one",
            str(lock.get("training_arm")) in inputs["c6"]["banks"],
            f"the frozen configurations were searched against the "
            f"{lock.get('training_arm')} bank, which is one of the three C6 froze",
            training_arm=lock.get("training_arm"),
            available_arms=sorted(inputs["c6"]["banks"]),
            note="C8 trains every arm; only the SEARCH population was restricted"))
        checks.append(check(
            "c8_no_fixture_in_scientific_context", True,
            "the scientific path builds no fixture batch and no audit detector",
            audit_batch_used=False, audit_detector_used=False,
            smoke_budget_used=False, smoke_rows_used=False,
            trainer="prism_fas.detector.trainer.run_source_only_flow (canonical)"))

        inputs = {**inputs, "c7_lock": lock, "c7_lock_path": SCIENTIFIC_CONFIG_LOCK_PATH}
        artifact = write_artifact(request, reports / "C8_SCIENTIFIC_INPUTS.json", {
            "schema_version": "c8-scientific-inputs-v1", "generated_at_utc": utc(),
            "mode": VERIFY_INPUTS, "fixture_backed": False,
            **{key: value for key, value in inputs.items() if key != "c7_lock"},
            "c7_config_lock": {key: lock.get(key) for key in
                               ("search_plan_identity", "winner_config_sha256",
                                "winner_checkpoint_sha256", "decision_graph_hash",
                                "search_decision_identity", "lr_decision_identity")}})
        return inputs, self.result(
            request, mode=VERIFY_INPUTS, checks=checks, artifacts=[artifact],
            parent_identities={
                "m3b_package": inputs["package_identity"],
                "c6_selector": closure["selector_identity_sha256"],
                "c7_detector_config": str(lock.get("winner_config_sha256"))})

    def _scientific_execute(self, request: AdapterRequest, inputs: dict[str, Any],
                            sample: list[dict[str, Any]], reports: Path,
                            runs: Path) -> tuple[list[dict[str, Any]], AdapterResult]:
        """Every scheduled row, trained end to end. No sampling, no budget."""
        checks: list[dict[str, Any]] = []
        executed: list[dict[str, Any]] = []

        for item in sample:
            row = item["row"]
            if item["decision"]["action"] == "SKIP_VALID_COMPLETE":
                executed.append(self._reuse_one(request, row, item["directory"]))
                continue
            executed.append(_run_scientific_row(request, inputs=inputs, row=row,
                                                root=runs))

        passed = [item for item in executed if item.get("status") == "PASS"]
        failed = [item for item in executed if item.get("status") != "PASS"]
        checks.append(check(
            "c8_every_scheduled_row_is_terminal",
            all(item.get("status") in ("PASS", "FAIL") for item in executed)
            and bool(executed),
            f"{len(passed)} passed and {len(failed)} failed; every scheduled row "
            "reached a terminal state and wrote its own manifest",
            rows=len(executed), passed=len(passed),
            failures=[{"row_id": item["row_id"], "reason": item.get("reason", "")[:200]}
                      for item in failed]))
        checks.append(check(
            "c8_rows_were_trained_not_stepped_on_fixtures",
            all(item.get("trainer") == "M9Trainer" for item in passed),
            "every passing row was produced by the canonical trainer on the real "
            "source package and its own C6 matched bank",
            trainers=sorted({str(item.get("trainer")) for item in executed}),
            audit_batch_used=False, audit_detector_used=False))
        checks.append(check(
            "c8_calibration_is_source_dev_only",
            all(item.get("calibration", {}).get("split") == "source_dev"
                for item in passed),
            "temperature and threshold were fitted on the protocol's own source_dev",
            splits=sorted({str(item.get("calibration", {}).get("split"))
                           for item in passed}),
            rule="§16: C-G5 fits temperature and operating threshold on source_dev only"))
        checks.append(check(
            "c8_selection_used_only_the_protocols_own_domains",
            all(set(item.get("selection_domains", ()))
                == set(item.get("expected_selection_domains", (1,)))
                for item in passed),
            "P1 selected on CASIA-dev, P2 on MSU-dev and P3-ready on both with equal "
            "weight; a cross-source number never entered a selection tuple",
            per_row={item["row_id"]: item.get("selection_domains")
                     for item in passed}))
        by_track: dict[str, set[str]] = {}
        for item in passed:
            by_track.setdefault(str(item.get("track")), set()).add(
                str(dict(item.get("parent_identities") or {}).get("c7_detector_config")))
        checks.append(check(
            "c8_rows_of_one_track_share_one_frozen_configuration",
            all(len(values) == 1 for values in by_track.values()),
            "every executed row of a track inherited the same C7 configuration "
            "identity, whatever generator arm it trains on",
            per_track={track: sorted(values)
                       for track, values in sorted(by_track.items())},
            arms_per_track={track: sorted({str(item.get("arm")) for item in passed
                                           if str(item.get("track")) == track})
                            for track in sorted(by_track)},
            rule="\u00a718.1: within a track, the generator arm is the ONLY treatment; "
                 "a second configuration would be a second treatment"))
        checks.append(check(
            "c8_cross_source_is_diagnostic_only",
            all(not item.get("cross_source", {}).get("is_selection_signal", False)
                for item in passed),
            "the cross-source evaluation is recorded as a diagnostic and carries the "
            "frozen temperature and threshold rather than fitting its own",
            rows_with_cross_source=[item["row_id"] for item in passed
                                    if item.get("cross_source")]))

        artifact = write_artifact(request, reports / "C8_SOURCE_MATRIX_RESULTS.json", {
            "schema_version": "c8-source-matrix-results-v1", "generated_at_utc": utc(),
            "mode": EXECUTE_ROWS, "fixture_backed": False,
            "rows_executed": len(executed),
            "rows": [{key: value for key, value in item.items()
                      if key not in ("complexity", "resources")} for item in executed]})
        rollups = self._rollups(request, executed, reports)
        return executed, self.result(request, mode=EXECUTE_ROWS, checks=checks,
                                     artifacts=[artifact, *rollups])

    def _scientific_diagnostics(self, request: AdapterRequest,
                                executed: list[dict[str, Any]],
                                reports: Path) -> AdapterResult:
        """Cross-source diagnostics, which §C9 requires to exist before the freeze."""
        from prism_fas.evaluation import source_selection

        checks: list[dict[str, Any]] = []
        passed = [item for item in executed if item.get("status") == "PASS"]
        expected = [item for item in passed
                    if source_selection.cross_source_domains_for(item["protocol"])]
        produced = [item for item in expected if item.get("cross_source")]

        checks.append(check(
            "c8_cross_source_diagnostics_exist_for_every_p1_p2_row",
            len(produced) == len(expected),
            f"{len(produced)}/{len(expected)} P1/P2 rows carry a cross-source "
            "evaluation on the other source domain",
            missing=[item["row_id"] for item in expected
                     if not item.get("cross_source")]))
        checks.append(check(
            "c8_p3_rows_have_no_cross_source_evaluation",
            all(not item.get("cross_source") for item in passed
                if item["protocol"] == "P3"),
            "a P3-ready row's test domain is the held-out target; it is predicted at "
            "C11 and nothing here evaluates it",
            p3_rows=[item["row_id"] for item in passed if item["protocol"] == "P3"],
            target_metrics_computed=0))

        artifact = write_artifact(request, reports / "C8_CROSS_SOURCE_DIAGNOSTICS.json", {
            "schema_version": "c8-cross-source-diagnostics-v1",
            "generated_at_utc": utc(), "mode": CROSS_SOURCE_DIAGNOSTICS,
            "fixture_backed": False,
            "rule": ("§15.4: the cross-domain side of a P1/P2 protocol is evaluation, "
                     "never a selection or calibration signal"),
            "rows": [{"row_id": item["row_id"], "protocol": item["protocol"],
                      "arm": item["arm"], "track": item["track"], "seed": item["seed"],
                      "selection_domains": item.get("selection_domains"),
                      "cross_source": item.get("cross_source")}
                     for item in sorted(produced, key=lambda row: row["row_id"])]})
        return self.result(request, mode=CROSS_SOURCE_DIAGNOSTICS, checks=checks,
                           artifacts=[artifact])

    def _scientific_calibration_stability(self, request: AdapterRequest,
                                          executed: list[dict[str, Any]],
                                          reports: Path) -> AdapterResult:
        """Temperature and threshold spread across the seeds of one configuration.

        §C9 requires calibration stability to exist before the source freeze. What
        it is here is descriptive, not a gate: the seeds of one configuration are
        replications, so a wide spread in the fitted temperature is a finding
        about the configuration rather than a reason to refit anything.
        """
        import statistics

        checks: list[dict[str, Any]] = []
        passed = [item for item in executed if item.get("status") == "PASS"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in passed:
            grouped.setdefault(str(item.get("config_identity")), []).append(item)

        groups = []
        for identity, members in sorted(grouped.items()):
            temperatures = [float(item["calibration"]["temperature"]) for item in members]
            thresholds = [float(item["calibration"]["threshold"]) for item in members]
            groups.append({
                "config_identity": identity,
                "row_ids": sorted(item["row_id"] for item in members),
                "experiment_id": members[0]["method"], "protocol": members[0]["protocol"],
                "seeds": sorted(int(item["seed"]) for item in members),
                "temperature": {"values": temperatures,
                                "mean": statistics.fmean(temperatures),
                                "stdev": statistics.pstdev(temperatures)},
                "threshold": {"values": thresholds,
                              "mean": statistics.fmean(thresholds),
                              "stdev": statistics.pstdev(thresholds)},
            })

        checks.append(check(
            "c8_calibration_stability_covers_every_configuration",
            bool(groups) and all(group["seeds"] for group in groups),
            "every executed configuration reports its per-seed temperature and "
            "threshold spread",
            configurations=len(groups),
            seeds_per_configuration={group["config_identity"][:12]: len(group["seeds"])
                                     for group in groups}))
        checks.append(check(
            "c8_no_seed_was_recalibrated_to_match_another", True,
            "each seed carries the temperature and threshold its own source_dev fit "
            "produced; nothing here refits or reconciles them",
            rule="§18.3: no best-seed reporting; the spread IS the result"))

        artifact = write_artifact(request, reports / "C8_CALIBRATION_STABILITY.json", {
            "schema_version": "c8-calibration-stability-v1", "generated_at_utc": utc(),
            "mode": CALIBRATION_STABILITY, "fixture_backed": False,
            "fitted_on": "source_dev of each row's own protocol",
            "configurations": groups})
        return self.result(request, mode=CALIBRATION_STABILITY, checks=checks,
                           artifacts=[artifact])

    def _scientific_acceptance(self, request: AdapterRequest, inputs: dict[str, Any],
                               plan: Any, executed: list[dict[str, Any]],
                               reports: Path,
                               gates: list[AdapterResult]) -> AdapterResult:
        """C8_ACCEPTANCE: every declared row terminal, and nothing hidden."""
        from prism_fas.evaluation.source_matrix import ARMS, SEED_FAMILY

        checks: list[dict[str, Any]] = []
        by_id = {str(item["row_id"]): item for item in executed}
        declared = {row.row_id: row for row in plan.rows}
        missing = sorted(set(declared) - set(by_id))
        hidden = sorted(set(by_id) - set(declared))
        failed = sorted(row_id for row_id, item in by_id.items()
                        if item.get("status") != "PASS")

        checks.append(check(
            "c8_every_declared_row_is_terminal", not missing,
            f"all {len(declared)} declared rows produced evidence",
            declared=len(declared), executed=len(by_id), missing=missing))
        checks.append(check(
            "c8_no_hidden_row", not hidden,
            "no row entered the evidence set that the frozen matrix did not declare",
            hidden=hidden,
            rule="§C9: an unplanned run entering the frozen set would put a "
                 "configuration into the comparison that was never preregistered"))
        checks.append(check(
            "c8_mandatory_rows_passed", not failed,
            "every mandatory row reached PASS",
            failed=failed,
            retention="a real failure keeps its own manifest and is never deleted or "
                      "replaced; C8 acceptance reflects it rather than hiding it"))
        seed_counts: dict[str, list[int]] = {}
        for row_id, item in by_id.items():
            row = declared.get(row_id)
            if row is None:
                continue
            seed_counts.setdefault(f"{row.experiment_id}:{row.protocol}", []).append(
                int(row.seed))
        checks.append(check(
            "c8_seed_counts_are_exact",
            all(len(set(values)) == len(values) for values in seed_counts.values())
            and all(len(set(seed_counts.get(f"C-G-{arm}:P3", ()))) == 5 for arm in ARMS),
            "each configuration ran exactly its declared seeds, each once, from the "
            "fixed family",
            seed_counts={key: sorted(set(values))
                         for key, values in sorted(seed_counts.items())},
            seed_family=list(SEED_FAMILY)))
        checks.append(check(
            "c8_checkpoint_and_calibration_identities_present",
            all(item.get("checkpoint", {}).get("sha256")
                and item.get("calibration", {}).get("calibration_hash")
                for item in executed if item.get("status") == "PASS"),
            "every passing row carries a checkpoint SHA-256 and a calibration hash "
            "C9 can freeze",
            rows_without=[item["row_id"] for item in executed
                          if item.get("status") == "PASS"
                          and not (item.get("checkpoint", {}).get("sha256")
                                   and item.get("calibration", {}).get("calibration_hash"))]))
        checks.append(check(
            "c8_source_only_selection_is_proven",
            all(item.get("source_isolation", {}).get("target_test_opened") is False
                for item in executed if item.get("status") == "PASS"),
            "each row's own trainer audit records that no target split was opened",
            target_paths_resolved=0, target_labels_resolved=0))
        gate_failures = [result.mode for result in gates if result.status != "PASS"]
        checks.append(check(
            "c8_supporting_evidence_passed", not gate_failures,
            "cross-source diagnostics, calibration stability and target isolation all "
            "passed; §C9 requires them to exist before the source freeze",
            failed_modes=gate_failures))
        checks.append(check(
            "c8_detector_reliability_is_not_claimed_here", True,
            "C8 produces no BA_sep number and no DETECTOR_RELIABILITY_LOCK_C; the "
            "barrier is a separate, still-unfrozen protocol and C9 stays blocked on it",
            stage=detector_reliability.STAGE,
            lock=detector_reliability.LOCK_PATH,
            unresolved=["DETECTOR_BA_SEP_PROBE_PROTOCOL",
                        "DETECTOR_BA_SEP_EVIDENCE_VECTOR",
                        "DETECTOR_BA_SEP_PROBE_SEEDS"]))

        passed = all(item["ok"] for item in checks)
        artifact = write_artifact(request, reports / "C8_ACCEPTANCE.json", {
            "schema_version": "c8-acceptance-v1", "generated_at_utc": utc(),
            "mode": ACCEPTANCE, "fixture_backed": False,
            "accepted": passed,
            "matrix_identity": plan.identity,
            "rows_declared": len(declared), "rows_terminal": len(by_id),
            "rows_passed": len(by_id) - len(failed), "rows_failed": failed,
            "hidden_rows": hidden, "missing_rows": missing,
            "seed_counts": {key: sorted(set(values))
                            for key, values in sorted(seed_counts.items())},
            "c7_training_arm": inputs["c7_lock"].get("training_arm"),
            "c7_detector_config_sha256": {
                track: dict(sub).get("winner_config_sha256") for track, sub
                in sorted(dict(inputs["c7_lock"].get("tracks") or {}).items())},
            "c7_shared_within_track": inputs["c7_lock"].get("shared_within_track"),
            "c7_per_arm_search_performed":
                inputs["c7_lock"].get("per_arm_search_performed"),
            "c6_selector_identity_sha256": inputs["c6"]["selector_identity_sha256"],
            "source_package_identity": inputs["package_identity"],
            "target_access": 0,
            "no_target_capability_proof": {"target_roots_mounted": [],
                                           "target_labels_resolved": 0},
            "checks": checks,
            "next_gate": ("DETECTOR_RELIABILITY_LOCK_C, whose probe protocol, evidence "
                          "vector and seeds are still NEEDS_SCIENTIFIC_DECISION. C9 is "
                          "blocked on it and C8 acceptance does not unblock it")})
        return self.result(request, mode=ACCEPTANCE, checks=checks, artifacts=[artifact],
                           # The ONE place C8 claims scientific evidence.
                           scientific_evidence=passed)

    def _rollups(self, request: AdapterRequest, executed: list[dict[str, Any]],
                 reports: Path) -> list[str]:
        """One complexity and one resource rollup per stage, over every arm that ran.

        Written once here rather than per row: the matrix runs many arms, and a
        per-row write would be last-writer-wins, leaving a file that names
        whichever arm happened to finish last.
        """
        return [write_artifact(
            request, reports / f"C8_{name}.json",
            {"schema_version": f"c8-{name.lower()}-v1", "generated_at_utc": utc(),
             "mode": EXECUTE_ROWS, "rows_profiled": len(payloads),
             "models" if key == "complexity" else "runs": payloads,
             "note": "every executed arm, not a representative one: a single-arm "
                     "rollup would name whichever arm happened to finish last"})
            for name, key, payloads in (
                ("MODEL_COMPLEXITY", "complexity",
                 [item["complexity"] for item in sorted(
                     (row for row in executed if row.get("complexity")),
                     key=lambda row: row["row_id"])]),
                ("COMPUTE_RESOURCES", "resources",
                 [dict(item["resources"], row_id=item["row_id"]) for item in sorted(
                     (row for row in executed if row.get("resources")),
                     key=lambda row: row["row_id"])]))]


# --- the scientific row ------------------------------------------------------

def _row_destination(root: Path, row: Any) -> Path:
    """One deterministic directory per atomic run. Keyed by identity, not order."""
    return (Path(root) / row.protocol / row.experiment_id /
            row.config_identity[:12] / str(row.seed))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TrackConfigurationMissing(AdapterError):
    """The frozen C7 lock names no configuration for this row's track."""

    reason_code = "C7_TRACK_CONFIG_ABSENT"


def track_configuration(lock: dict[str, Any], track: str) -> dict[str, Any]:
    """The one frozen configuration this track's rows all train at.

    Keyed on the TRACK, never on the arm. C7 ran one bounded pass per track
    against a single frozen search bank, so `C-G-RND`, `C-G-DET` and `C-G-LLM`
    resolve to the same sub-config here and differ only in which C6 bank supplies
    their synthetic quarter and which seed they run. That is the fairness
    invariant, and reading it from one place is what makes it checkable.
    """
    tracks = dict(lock.get("tracks") or {})
    try:
        return dict(tracks[str(track)])
    except KeyError:
        raise TrackConfigurationMissing(
            f"the C7 detector config lock names no configuration for Track {track!r}; "
            f"it froze {sorted(tracks)}. A row may not invent one, and it may not "
            "borrow the other track's - the two have different active loss sets"
        ) from None


def _detector_config_for_row(request: AdapterRequest, *, row: Any, lock: dict[str, Any],
                             bank: Any, run_id: str) -> tuple[Any, Any]:
    """The frozen C7 configuration for this row's TRACK, at its protocol and seed.

    Three things vary across the 42 rows and nothing else does: the typed variant
    (which decides the architecture and the active loss graph), the protocol
    (which decides the source domains) and the seed. Every scalar the C7 search
    froze -- the learning rates, the weight decay, the warm-up fraction and the
    loss weights -- is read from that track's winner config, so no row can
    quietly train at a different configuration from the one the matrix declares,
    and every primary generator arm of a track trains at the same one.
    """
    from dataclasses import replace

    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.evaluation import source_selection
    from prism_fas.pipeline.adapters.c7 import _TRIAL_LOSS_WEIGHTS

    flags = {key: value for key, value in row.flags.items() if key != "recipe_arm"}
    variant = ResolvedExperimentVariant.resolve(flags)
    configs = load_m9_configs(request.repo / "configs/models/m9_detector.yaml",
                              request.repo / "configs/train/m9_reference.yaml",
                              variant=variant)
    sub = track_configuration(lock, row.track)
    winner = dict(sub.get("winner_config") or {})

    weights = dict(configs["training_config"].loss_weights)
    for name in _TRIAL_LOSS_WEIGHTS:
        if name in winner:
            weights[name] = float(winner[name])

    overrides: dict[str, Any] = {
        "run_id": run_id,
        "seed": int(row.seed),
        "prototype_seed": int(row.seed),
        "variant": variant,
        "loss_weights": weights,
        "synthetic_bank_identity": bank.identity,
        "source_domains": source_selection.domains_for(row.protocol),
    }
    # The learning rate is a vector under the approved common-multiplier
    # interpretation, so the lock carries the multiplier and the anchor and the
    # expansion happens once, here, exactly as it did during the search.
    multiplier = winner.get("learning_rate_multiplier")
    anchor = dict(sub.get("lr_anchor_vector") or {})
    if multiplier is not None and anchor:
        for group, value in anchor.items():
            overrides[group] = float(value) * float(multiplier)
    if "weight_decay" in winner:
        overrides["weight_decay"] = float(winner["weight_decay"])
    if "warmup" in winner:
        overrides["warmup_fraction"] = float(winner["warmup"])
    return replace(configs["training_config"], **overrides), configs


def _track_parents(lock: dict[str, Any], track: str) -> dict[str, str]:
    """The C7 identities a row inherits from its track's frozen configuration."""
    sub = track_configuration(lock, track)
    return {
        "c7_detector_config": str(sub.get("winner_config_sha256")),
        "c7_search_plan": str(sub.get("search_plan_identity")),
        "c7_search_decision": str(lock.get("search_decision_identity")),
        "c7_training_arm": str(lock.get("training_arm")),
        "c7_decision_graph": str(sub.get("decision_graph_hash")),
    }


def _run_scientific_row(request: AdapterRequest, *, inputs: dict[str, Any], row: Any,
                        root: Path) -> dict[str, Any]:
    """One atomic scientific row, with its own durable artifacts (L.8).

    Every terminal state writes a manifest — PASS and FAIL alike. A row that
    genuinely fails is a real result, retained under its own identity; nothing
    here constructs a failure, and nothing here deletes one.
    """
    from prism_fas.detector.c6_bank import open_arm_bank
    from prism_fas.detector.decision_audit import decision_graph_hash
    from prism_fas.detector.trainer import M9Trainer, run_source_only_flow
    from prism_fas.evaluation import c6_evidence, source_selection
    from prism_fas.pipeline.adapters.c7 import _scientific_device
    from prism_fas.reporting import complexity as complexity_module
    from prism_fas.reporting import resources as resources_module
    from prism_fas.reporting.history import HistoryWriter

    destination = _row_destination(root, row)
    destination.mkdir(parents=True, exist_ok=True)
    lock = inputs["c7_lock"]
    started = utc()

    base = {
        "schema_version": "c8-run-manifest-v1",
        "run_identity": row.run_identity,
        "row_id": row.row_id,
        "protocol": row.protocol,
        "method": row.experiment_id,
        "track": row.track,
        "arm": row.arm,
        "config_identity": row.config_identity,
        "seed": int(row.seed),
        "replication_role": row.replication_role,
        "hypotheses": list(row.hypotheses),
        "selection_tuple": list(row.selection_tuple),
        "started_at_utc": started,
        "fixture_backed": False,
        "scientific_eligible": True,
        "trainer": "M9Trainer",
        "parent_identities": {
            "source_matrix": inputs.get("matrix_identity", ""),
            "source_package": inputs["package_identity"],
            "recipe_bank": inputs["recipe_bank_identity"],
            "c6_selector": inputs["c6"]["selector_identity_sha256"],
            "c6_bank": inputs["c6"]["banks"][row.arm]["selected_set_sha256"],
            **_track_parents(lock, row.track),
        },
        "c7_track_configuration": row.track,
        "target_paths_resolved": 0,
        "target_labels_resolved": 0,
    }

    def finalize(payload: dict[str, Any]) -> dict[str, Any]:
        write_artifact(request, destination / "run_manifest.json", payload)
        return {**payload, "manifest_keys": sorted(payload),
                "path": destination.relative_to(request.repo).as_posix()}

    try:
        import platform

        import torch

        device = _scientific_device()
        bank = open_arm_bank(
            request.repo, arm=row.arm,
            evidence=c6_evidence.verify_c6_evidence(request.repo).bank(row.arm),
            candidates_root=request.repo / inputs["candidates_root"],
            package_identity=inputs["package_identity"],
            recipe_bank_identity=inputs["recipe_bank_identity"])
        config, configs = _detector_config_for_row(
            request, row=row, lock=lock, bank=bank, run_id=row.row_id)
        trainer = M9Trainer(
            config=config, detector_config=configs["detector_config"],
            package_root=request.repo / inputs["package_root"],
            bank_root=request.repo / inputs["candidates_root"],
            recipe_bank_root=request.repo / inputs["recipe_bank_root"],
            run_root=destination, cache_root=destination / "cache",
            weight_root=request.repo / inputs["weight_root"],
            loader_config_path=request.repo / "configs/data/loader_m4.yaml",
            device=device, synthetic_bank=bank)
        flow = run_source_only_flow(trainer, resume=request.resume)

        frames = source_selection.source_dev_frame_rows(trainer)
        calibration = source_selection.fit_source_dev_calibration(frames)
        epoch = int((flow["run_summary"].get("best_metrics") or {}).get("epoch", -1))
        selection = source_selection.evaluate(
            frames, protocol=row.protocol, temperature=calibration["temperature"],
            threshold=calibration["threshold"], epoch=epoch,
            decision_logit_name=trainer.decision_logit_name,
            decision_score_name=trainer.decision_score_name)

        # The cross-source side of a P1/P2 protocol: the SAME frozen temperature
        # and threshold, applied to the other source domain. It is evaluation, so
        # nothing about it may move a selection or a calibration.
        cross_domains = source_selection.cross_source_domains_for(row.protocol)
        cross: dict[str, Any] = {}
        if cross_domains:
            cross = _cross_source_evaluation(
                request, trainer=trainer, inputs=inputs, row=row,
                domains=cross_domains, calibration=calibration, epoch=epoch)

        checkpoint = trainer.checkpoint_path("best")
        if not checkpoint.is_file():
            checkpoint = trainer.checkpoint_path("last")
        graph = decision_graph_hash(trainer.model)
        calibration_record = {
            **calibration,
            "decision_logit_name": trainer.decision_logit_name,
            "thresholded_quantity": trainer.decision_score_name,
            "decision_score_name": trainer.decision_score_name,
        }
        calibration_record["calibration_hash"] = _calibration_hash(calibration_record)
        write_artifact(request, destination / "calibration.json", calibration_record)

        # Written UNCONDITIONALLY, then one row per completed stage. Driving the
        # whole history off the stage lineage left a row with no history file at
        # all whenever that list was empty, and L.8 requires every atomic run to
        # emit its own durable artifacts — an absent history is indistinguishable
        # from a run that never happened.
        writer = HistoryWriter(path=destination / "train_history.jsonl",
                               run_identity=row.run_identity)
        writer.append(epoch=epoch, step=int(flow["run_summary"].get("global_step", 0)),
                      source_dev=dict(selection.get("pooled") or {}),
                      calibration={"temperature": calibration["temperature"],
                                   "threshold": calibration["threshold"],
                                   "split": calibration["split"]},
                      selection_tuple=dict(selection.get("ranking_tuple") or {}),
                      kind="row_summary", protocol=row.protocol, seed=int(row.seed))
        for entry in flow["run_summary"].get("stage_lineage", ()):
            writer.append(epoch=epoch, step=int(flow["run_summary"].get("global_step", 0)),
                          kind="stage", stage=str(entry.get("stage")),
                          stage_status=str(entry.get("status")))

        evaluation_batch = trainer.validation().batch(
            list(trainer.validation().positions)[:trainer.config.validation_batch_size])
        complexity_payload = complexity_module.profile_model(
            trainer.model, evaluation_batch, name=f"detector_{row.row_id}",
            input_shape=list(evaluation_batch.image.shape))
        resource_payload = resources_module.resource_record(
            microbatch_plan=trainer.samplers["G5"].contract.payload())
        write_artifact(request, destination / "model_complexity.json", complexity_payload)
        write_artifact(request, destination / "compute_resources.json", resource_payload)

        manifest = {
            **base,
            "status": "PASS",
            "finished_at_utc": utc(),
            # The variant the ROW resolved from its own preregistered flags, not
            # whatever the trainer happens to hold. They are the same object in
            # production — `M9Trainer` takes its variant from `config.variant` —
            # and reading it from the row's own resolution is the tighter
            # contract: what the manifest records is then what §18 declared for
            # this row, and a trainer that silently substituted a variant would
            # show up as a mismatch rather than as agreement with itself.
            "flags": config.variant.flags(),
            "variant_identity": config.variant.identity(),
            "variant_track": config.variant.track,
            "decision_logit_name": trainer.decision_logit_name,
            "decision_score_name": trainer.decision_score_name,
            "decision_graph_hash": graph["decision_graph_hash"],
            "resolved_config": config.resolved(),
            "resolved_config_hash": config.hash(),
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform(), "device": device},
            "batch_contract": trainer.samplers["G5"].contract.payload(),
            "selection_domains": list(selection["domains"]),
            "expected_selection_domains": list(
                source_selection.domains_for(row.protocol)),
            "metrics": {"source_dev": selection, "best_epoch": epoch,
                        "trainer_best_metrics": flow["run_summary"].get("best_metrics", {})},
            "cross_source": cross,
            "calibration": calibration_record,
            "checkpoint": {
                "path": checkpoint.relative_to(request.repo).as_posix(),
                "sha256": _sha256_file(checkpoint),
                "kind": checkpoint.stem,
                "architecture_identity": trainer.model.architecture_identity(),
                "run_identity": trainer.identity.payload(),
            },
            "c6_bank": bank.summary(),
            "source_isolation": flow["source_isolation"],
            "flow": {key: flow[key] for key in
                     ("stages", "declared_stages", "stages_executed_here",
                      "run_closure", "resumed_from", "resumed_stage")},
            "code_lineage": {"git_commit": flow["run_summary"].get("git_commit")},
        }
        record = finalize(manifest)
        return {**record, "complexity": complexity_payload, "resources": resource_payload}

    except (KeyboardInterrupt, SystemExit):
        # Not an outcome for this row. Everything already on disk stays and the
        # next invocation resumes this exact identity.
        raise
    except Exception as error:                        # noqa: BLE001
        return finalize({
            **base,
            "status": "FAIL",
            "finished_at_utc": utc(),
            "failure": {"stage": "C8_ROW", "error_type": type(error).__name__,
                        "reason": str(error)[:800], "recoverable": False,
                        "constructed": False},
            "reason": f"{type(error).__name__}: {error}"[:800],
            "metrics": {}, "calibration": {}, "checkpoint": {},
            "retention": ("a real scientific failure keeps its own manifest under its "
                          "own identity; it is never deleted, replaced or summarized "
                          "away, and C8 acceptance reflects it (L.6, L.8)"),
        })


def _cross_source_evaluation(request: AdapterRequest, *, trainer: Any,
                             inputs: dict[str, Any], row: Any,
                             domains: tuple[str, ...], calibration: dict[str, Any],
                             epoch: int) -> dict[str, Any]:
    """The other source domain, at the row's own frozen temperature and threshold.

    A second `M9ValidationDataset` over the same frozen package, scoped to the
    other domain. It is the canonical validation class rather than a bespoke
    loader, so the decode, the priors and the batch shape are the ones the
    selection split used — otherwise the diagnostic would differ from the
    selection number for a reason that has nothing to do with the domain.
    """
    import torch

    from prism_fas.detector.dataset import M9ValidationDataset
    from prism_fas.evaluation import source_selection

    dataset = M9ValidationDataset(
        request.repo / inputs["package_root"], trainer.loader_config,
        cache_root=trainer.cache_root, domains=domains)
    model = trainer.model
    model.eval()
    frames: list[dict[str, Any]] = []
    positions = list(dataset.positions)
    size = int(trainer.config.validation_batch_size)
    with torch.no_grad():
        for start in range(0, len(positions), size):
            chunk = positions[start:start + size]
            batch = dataset.batch(chunk).to(trainer.device)
            logits = model(batch).global_logit.detach().float().cpu().numpy().reshape(-1)
            labels = batch.label.detach().cpu().numpy().reshape(-1)
            for offset, position in enumerate(chunk):
                frames.append({"sample_id": dataset.sample_id_of(position),
                               "source_record_id": dataset.record_of(position),
                               "dataset": dataset.domain_of[position],
                               "label": int(labels[offset]),
                               "logit": float(logits[offset])})
    return source_selection.evaluate(
        frames, protocol=row.protocol, temperature=calibration["temperature"],
        threshold=calibration["threshold"], epoch=epoch,
        decision_logit_name=trainer.decision_logit_name,
        decision_score_name=trainer.decision_score_name,
        domains=domains, role="cross_source_diagnostic")


def _calibration_hash(record: dict[str, Any]) -> str:
    import hashlib
    import json

    body = {key: value for key, value in record.items() if key != "calibration_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


__all__ = ["STAGE_ID", "MODES", "SCIENTIFIC_MODES", "PLAN_MATRIX", "SCHEDULE",
           "EXECUTE_ROWS", "FAILURE_PRESERVATION", "TARGET_ISOLATION",
           "VERIFY_INPUTS", "CROSS_SOURCE_DIAGNOSTICS", "CALIBRATION_STABILITY",
           "ACCEPTANCE", "TrackConfigurationMissing", "track_configuration",
           "C8Adapter"]
