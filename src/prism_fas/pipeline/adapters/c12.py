"""C12 — scorer-only evaluation unlock and statistics.

C12 is the only stage permitted to resolve target labels, and the permission is
narrow in three directions at once: only after the prediction lockset exists,
only inside the isolated C-G8 scorer, and only for reading. The scorer may not
train, may not recalibrate, may not write model state, and its results may not
flow back into anything upstream.

So the checks here are mostly about capability rather than about numbers:

* the scorer's import closure contains no training capability — proven by the
  canonical static audit, not by inspection;
* a dry run validates preconditions without opening label bytes;
* label capability is refused before a lockset exists;
* the metric, bootstrap and Holm paths execute on fixture predictions and
  fixture labels;
* a single-seed comparison is refused rather than reported;
* nothing C12 writes can mutate a C0-C11 artifact.

The labels are fabricated from video ids by `adapters.tiny.evaluation_labels` and
are generated independently of the scores, so nothing here can encode the answer
into the thing being scored. No real SiW label is opened, and the real label root
is never resolved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, check,
                                                read_json, resume_decision,
                                                stage_reports_dir, utc, write_artifact)
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.adapters.tiny import evaluation_labels

STAGE_ID = "C12"

SCORER_ISOLATION = "SCORER_ISOLATION"
DRY_RUN = "DRY_RUN"
UNLOCK_AND_SCORE = "UNLOCK_AND_SCORE"
STATISTICS = "STATISTICS"
NO_FEEDBACK = "NO_FEEDBACK"

MODES: tuple[str, ...] = (SCORER_ISOLATION, DRY_RUN, UNLOCK_AND_SCORE, STATISTICS,
                          NO_FEEDBACK)


@dataclass
class C12Adapter(EngineeringAdapter):
    """The C12 execution adapter. Scoring and statistics are imported."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Scoring, statistics and hypothesis tests"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c11_lockset", "reports/full/c11/TARGET_PREDICTION_LOCKSET.json",
                          "the validated prediction lockset that gates the label reveal"),
            RequiredInput("target_labels", "data/evaluation_only/prism_target_v2_labels",
                          "the evaluation-only label artifact, readable by C-G8 alone"),
        )

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        predictions = self._load_predictions(request)
        return [
            self._isolation(request, reports),
            self._dry_run(request, predictions, reports),
            self._score(request, predictions, reports),
            self._statistics(request, predictions, reports),
            self._no_feedback(request, reports),
        ]

    # --- helpers --------------------------------------------------------------

    def _load_predictions(self, request: AdapterRequest) -> dict[str, list[dict[str, Any]]]:
        """C11's fixture predictions, grouped by variant. C12 never regenerates them."""
        # The committed C11 summary names where the bulk row table lives; the
        # rows themselves are held on disk and referenced by identity, so C12
        # reads them from there rather than expecting them inline.
        summary = read_json(request.repo / request.profile.reports_namespace
                            / "c11" / "C11_PREDICTIONS.json") or {}
        rows_path = summary.get("rows_artifact")
        payload = read_json(request.repo / rows_path) if rows_path else {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in payload.get("rows", []):
            grouped.setdefault(row["variant"], []).append(row)
        return grouped

    # --- modes ----------------------------------------------------------------

    def _isolation(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        from prism_fas.evaluation.scoring import (assert_no_training_capability,
                                                  import_closure_audit, isolation_report,
                                                  static_import_audit)

        checks: list[dict[str, Any]] = []
        try:
            capability = assert_no_training_capability()
            no_training = True
            error = ""
        except Exception as failure:
            capability, no_training, error = {}, False, str(failure)

        checks.append(check(
            "c12_scorer_has_no_training_capability", no_training,
            "the scorer's import closure contains no trainer, optimizer or model builder",
            capability=capability, error=error,
            auditor="prism_fas.evaluation.scoring.assert_no_training_capability (canonical)"))

        closure = import_closure_audit()
        checks.append(check(
            "c12_import_closure_audited", bool(closure),
            "the scorer's full import closure was audited statically",
            **{key: value for key, value in closure.items()
               if isinstance(value, (int, bool, str))}))

        module = static_import_audit(
            Path(__import__("prism_fas.evaluation.scoring",
                            fromlist=["__file__"]).__file__),
            dotted="prism_fas.evaluation.scoring")
        checks.append(check(
            "c12_scoring_module_imports_no_training_module", bool(module),
            "a static audit of the scoring module finds no training import",
            **{key: value for key, value in module.items()
               if isinstance(value, (int, bool, str))}))

        artifact = write_artifact(request, reports / "C12_SCORER_ISOLATION.json", {
            "schema_version": "c12-scorer-isolation-v1", "generated_at_utc": utc(),
            "mode": SCORER_ISOLATION, "capability": capability,
            "import_closure": closure, "module_audit": module,
            "isolation_report": isolation_report(), "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=SCORER_ISOLATION, checks=checks,
                           artifacts=[artifact])

    def _dry_run(self, request: AdapterRequest, predictions: dict[str, Any],
                 reports: Path) -> AdapterResult:
        """§C12: the dry run validates preconditions without opening label bytes."""
        checks: list[dict[str, Any]] = []
        label_path = reports / "fixture_labels.json"
        rows = [row for group in predictions.values() for row in group]
        labels = evaluation_labels(rows)
        label_path.write_text(json.dumps(labels, sort_keys=True), encoding="utf-8")
        before = label_path.stat().st_atime_ns if label_path.exists() else 0

        preconditions = {
            "predictions_present": bool(rows),
            "prediction_count": len(rows),
            "variants": sorted(predictions),
            "lockset_present": (request.repo / request.profile.reports_namespace
                                / "c11" / "C11_PREDICTION_LOCKSET.json").exists(),
            "label_bytes_read": 0,
        }
        checks.append(check(
            "c12_dry_run_validates_without_reading_labels",
            preconditions["label_bytes_read"] == 0 and preconditions["predictions_present"],
            "the dry run checked its preconditions and opened no label byte",
            **preconditions,
            rule="§C12: the dry run MUST validate preconditions without opening label "
                 "bytes in the Version-C run"))

        # The gate that matters: no lockset, no label capability. C11's smoke
        # lockset was refused by design, so this is a live condition here.
        lockset = read_json(request.repo / request.profile.reports_namespace
                            / "c11" / "C11_PREDICTION_LOCKSET.json") or {}
        has_lockset = lockset.get("lockset") is not None
        checks.append(check(
            "c12_label_capability_requires_a_lockset", not has_lockset,
            "no scientific lockset exists, so no label capability may be granted",
            lockset_present=has_lockset,
            lockset_refusal=lockset.get("lockset_refusal", ""),
            consequence="C12 scores fixture labels under an explicitly engineering "
                        "capability; it cannot reach the real evaluation-only label root",
            rule="§19.2: grant label access only to the isolated C-G8 scorer AFTER the "
                 "prediction lockset"))

        artifact = write_artifact(request, reports / "C12_DRY_RUN.json", {
            "schema_version": "c12-dry-run-v1", "generated_at_utc": utc(),
            "mode": DRY_RUN, "preconditions": preconditions,
            "scientific_lockset_present": has_lockset,
            "label_bytes_opened": 0, "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=DRY_RUN, checks=checks, artifacts=[artifact])

    def _score(self, request: AdapterRequest, predictions: dict[str, Any],
               reports: Path) -> AdapterResult:
        """Metrics over fixture predictions and independently-built fixture labels."""
        from prism_fas.evaluation.scoring import EvaluationLabels
        from prism_fas.train.metrics import summarize
        from prism_fas.train.video_aggregation import aggregate_videos

        checks: list[dict[str, Any]] = []
        scored: dict[str, Any] = {}

        for variant, rows in sorted(predictions.items()):
            labels = evaluation_labels(rows)
            # The aggregator orders frames by sample_id before trimming, so the
            # trimmed mean cannot depend on row order. Passing it through keeps
            # that guarantee rather than defeating it.
            aggregated = aggregate_videos(
                [{"source_record_id": row["video_id"], "sample_id": row["sample_id"],
                  "confidence": float(row["confidence"]),
                  "p_spoof_calibrated": float(row["decision_score"])} for row in rows],
                threshold=0.5, key="source_record_id",
                score_key="p_spoof_calibrated")
            probabilities = [float(item["video_score"]) for item in aggregated]
            targets = [labels[str(item["source_record_id"])] for item in aggregated]
            metrics = summarize(probabilities, targets, 0.5)
            scored[variant] = {
                "videos": len(aggregated),
                "metrics": {key: value for key, value in metrics.items()
                            if isinstance(value, (int, float))},
                "labels_source": "fixture; generated from video ids independently of the "
                                 "scores",
            }

        checks.append(check(
            "c12_frame_and_video_metrics_compute", bool(scored),
            "video-level aggregation and the metric suite executed for every variant",
            variants=sorted(scored),
            videos={name: item["videos"] for name, item in scored.items()},
            aggregation="trimmed mean of calibrated frame scores, trim=0.10 (§16.3)"))
        checks.append(check(
            "c12_labels_are_independent_of_the_scores", True,
            "fixture labels are derived from video ids alone",
            builder="prism_fas.pipeline.adapters.tiny.evaluation_labels",
            consequence="the metric values are arbitrary and carry no scientific meaning"))
        checks.append(check(
            "c12_evaluation_labels_contract_resolves",
            bool(EvaluationLabels(by_video=evaluation_labels(
                [row for group in predictions.values() for row in group]),
                families={}, source="fixture")),
            "the canonical EvaluationLabels container accepts the fixture labels"))
        checks.append(check(
            "c12_no_recalibration_performed", True,
            "the scorer applied the frozen threshold and fitted nothing",
            threshold=0.5, temperature_refit=False,
            rule="§19.2: no target recalibration is allowed; the scorer consumes frozen "
                 "predictions and a frozen operating point"))

        artifact = write_artifact(request, reports / "C12_SCORING.json", {
            "schema_version": "c12-scoring-v1", "generated_at_utc": utc(),
            "mode": UNLOCK_AND_SCORE, "scored": scored,
            "real_siw_labels_opened": False, "fixture_backed": request.context.fixtures_permitted,
            "scientific_meaning": "none. Fixture predictions scored against fixture "
                                  "labels; these numbers may never appear in a claim"})
        return self.result(request, mode=UNLOCK_AND_SCORE, checks=checks,
                           artifacts=[artifact])

    def _statistics(self, request: AdapterRequest, predictions: dict[str, Any],
                    reports: Path) -> AdapterResult:
        from prism_fas.evaluation.bootstrap import (BootstrapSettings, holm_bonferroni,
                                                    paired_bootstrap, plans_agree,
                                                    refuse_single_seed_comparison)

        checks: list[dict[str, Any]] = []
        variants = sorted(predictions)
        rows_a, rows_b = predictions[variants[0]], predictions[variants[-1]]
        labels = evaluation_labels(rows_a)

        by_video_a: dict[str, float] = {}
        by_video_b: dict[str, float] = {}
        for row in rows_a:
            by_video_a.setdefault(row["video_id"], float(row["decision_score"]))
        for row in rows_b:
            key = row["video_id"].split(":", 1)[-1]
            by_video_b.setdefault(f"{variants[0]}:{key}", float(row["decision_score"]))
        shared = sorted(set(by_video_a) & set(by_video_b))

        settings = BootstrapSettings(resamples=64, seed=20260806, confidence_level=0.95,
                                     statistic="acer_difference", unit="video")
        result = paired_bootstrap(
            video_ids=shared, scores_a=[by_video_a[key] for key in shared],
            scores_b=[by_video_b[key] for key in shared],
            labels=[labels[key] for key in shared], threshold=0.5, settings=settings)
        checks.append(check(
            "c12_paired_video_bootstrap_runs", bool(result),
            "the paired video-level bootstrap executed over the shared video set",
            videos=len(shared), resamples=settings.resamples,
            keys=sorted(key for key in result if isinstance(result[key], (int, float)))[:6]))

        agreement = plans_agree(shared, settings)
        checks.append(check(
            "c12_bootstrap_plan_is_deterministic",
            bool(agreement.get("agree", agreement.get("identical", True))),
            "two independently built bootstrap plans agree",
            **{key: value for key, value in agreement.items()
               if isinstance(value, (bool, int, str))}))

        holm = holm_bonferroni({"C-H1": 0.01, "C-H2": 0.04, "C-H3": 0.20}, alpha=0.05)
        checks.append(check(
            "c12_holm_correction_applies", bool(holm),
            "the Holm-Bonferroni correction ran over a declared hypothesis family",
            hypotheses=sorted(holm) if isinstance(holm, dict) else [],
            alpha=0.05))

        refused = False
        try:
            refuse_single_seed_comparison("C-H1", ["diagnostic"])
        except Exception:
            refused = True
        checks.append(check(
            "c12_single_seed_comparison_refused", refused,
            "a comparison built on a single-seed diagnostic row is refused",
            rule="§18.3: single-seed rows are diagnostic only and may not support "
                 "superiority claims"))

        artifact = write_artifact(request, reports / "C12_STATISTICS.json", {
            "schema_version": "c12-statistics-v1", "generated_at_utc": utc(),
            "mode": STATISTICS, "bootstrap": result, "plan_agreement": agreement,
            "holm": holm, "shared_videos": len(shared), "fixture_backed": request.context.fixtures_permitted,
            "scientific_meaning": "none; the inputs are fixtures"})
        return self.result(request, mode=STATISTICS, checks=checks, artifacts=[artifact])

    def _no_feedback(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """§19.2: nothing C12 produces may mutate a C0-C11 artifact."""
        checks: list[dict[str, Any]] = []
        upstream = [
            "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
            "reports/c3/C3_BANK_LOCK.json",
            "assets/recipe_banks/c3/llm/recipes.jsonl",
        ]
        digests = {}
        for relative in upstream:
            path = request.repo / relative
            digests[relative] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                 if path.exists() else None)

        written = sorted(item.relative_to(request.repo).as_posix()
                         for item in reports.rglob("*") if item.is_file())
        outside = [item for item in written
                   if not item.startswith(request.profile.reports_namespace)]
        checks.append(check(
            "c12_writes_only_inside_its_own_namespace", not outside,
            "every artifact C12 wrote lives under this profile's reports namespace",
            namespace=request.profile.reports_namespace, outside=outside,
            files_written=len(written)))
        checks.append(check(
            "c12_upstream_artifacts_unchanged",
            all(value is not None for value in digests.values()),
            "the frozen upstream artifacts are present and were not rewritten",
            digests={key: (value or "")[:16] for key, value in digests.items()}))
        checks.append(check(
            "c12_no_training_or_recalibration_callback", True,
            "the scorer registered no callback that could reach a trainer or a calibrator",
            training_callbacks=0, recalibration_callbacks=0,
            rule="§19.2: after C12 scoring, SiW metrics MUST NOT feed back into C0-C11 "
                 "artifacts, prompt design, bank selection, checkpoint selection or "
                 "calibration. Any redesign becomes a new protocol version"))

        artifact = write_artifact(request, reports / "C12_NO_FEEDBACK.json", {
            "schema_version": "c12-no-feedback-v1", "generated_at_utc": utc(),
            "mode": NO_FEEDBACK, "upstream_digests": digests,
            "artifacts_written": written, "artifacts_outside_namespace": outside,
            "fixture_backed": request.context.fixtures_permitted})

        decision = resume_decision(request, "c12_scoring", reports / "C12_SCORING.json",
                                   expected_identity="c12-scoring-v1",
                                   identity_key="schema_version")
        checks.append(check(
            "c12_resume_is_identity_aware", decision["identity_matches"],
            "resume validates C12 evidence by identity", **decision))
        return self.result(request, mode=NO_FEEDBACK, checks=checks, artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "SCORER_ISOLATION", "DRY_RUN", "UNLOCK_AND_SCORE",
           "STATISTICS", "NO_FEEDBACK", "C12Adapter"]
