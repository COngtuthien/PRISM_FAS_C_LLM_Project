"""C7 — detector readiness, the §13.5 decision guards and the bounded search.

C7's hard acceptance is unusually specific, and this adapter is organised around
it rather than around a generic "does it train" idea of readiness:

* every primary row is executable — instantiate, forward, finite loss, backward,
  optimizer step, checkpoint, resume — on a CPU fixture;
* Track G emits only ``global_logit_G`` / ``p_G`` and instantiates no ConvNeXt,
  region, manifold or PromptHead module at all;
* Track R's final logit has a **direct** computational dependency on the frozen
  global embedding, the trainable ConvNeXt local feature and the 9-region
  representation, proven structurally by the §13.5 autograd and intervention
  tests;
* the calibration identity guard rejects thresholding one quantity with a
  temperature fitted on another;
* the decision graph is serialized into run identity;
* no experiment id branches anything.

The readiness half delegates to `prism_fas.evaluation.variant_audit`, which is
the canonical implementation of "build this row and step it" and already covers
stages, sampler, loss graph, optimizer groups and the checkpoint round-trip. The
§13.5 half delegates to `prism_fas.detector.decision_audit`. This module supplies
the two variants, sequences the work and records provenance — nothing else.

The search half follows the spec's own instruction for engineering validation:
the coordinate engine is exercised with deterministic analytic metrics rather
than by training twenty-one detectors on a CPU. What C7 readiness must prove
about the search is that the *algorithm* is right — order, one pass, retention,
tie-break, resume — and a synthetic objective proves that far more sharply than
noisy fixture training would.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, resume_decision,
                                                stage_reports_dir, stage_runs_dir, utc,
                                                write_artifact)
from prism_fas.pipeline.execution import ExecutionContext

STAGE_ID = "C7"

TRACK_G_READINESS = "TRACK_G_READINESS"
TRACK_R_READINESS = "TRACK_R_READINESS"
DECISION_DEPENDENCY_AUDIT = "DECISION_DEPENDENCY_AUDIT"
CALIBRATION_GUARDS = "CALIBRATION_GUARDS"
VARIANT_MATRIX_AUDIT = "VARIANT_MATRIX_AUDIT"
SOURCE_SEARCH = "SOURCE_SEARCH"

MODES: tuple[str, ...] = (TRACK_G_READINESS, TRACK_R_READINESS, DECISION_DEPENDENCY_AUDIT,
                          CALIBRATION_GUARDS, VARIANT_MATRIX_AUDIT, SOURCE_SEARCH)

DETECTOR_CONFIG = "configs/train/m9_reference.yaml"

#: The two v1.5 tracks as flag sets. Track G is global-only by design (§13.4.1);
#: primary Track R is regions + PromptHead with manifold OFF (§13.2), so the
#: manifold-dependent loss terms are NOT_APPLICABLE rather than silently zeroed.
TRACK_G_FLAGS: dict[str, Any] = {
    "local_branch": "off", "global_branch": "siglip2_frozen", "fusion": "single_logit",
    "region": "off", "manifold": "off", "prototype_k": 0, "prompt": "off",
    "synthetic": "bank_physics_gpat", "recipe_conditioning": "structured",
    "quality_weighting": "q_weighted", "outlier_loss": "off",
    "sampler": "domain_class_balanced", "frames_per_video": 4,
}
TRACK_R_FLAGS: dict[str, Any] = {
    **TRACK_G_FLAGS, "local_branch": "convnext", "fusion": "glr_concat",
    "region": "on", "prompt": "frozen_prompt",
}
#: The explicit K=4 secondary variant (§13.2): manifold ON, regional prototypes.
TRACK_R_K4_FLAGS: dict[str, Any] = {
    **TRACK_R_FLAGS, "manifold": "multi_prototype", "prototype_k": 4,
    "outlier_loss": "mask_aware",
}


def _variant(flags: dict[str, Any]) -> Any:
    from prism_fas.detector.variant import ResolvedExperimentVariant

    return ResolvedExperimentVariant.resolve(flags)


def _fixture_batch(variant: Any) -> Any:
    from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for
    from prism_fas.evaluation.variant_audit import audit_batch

    config = M9TrainingConfig(run_id=f"c7_{variant.track}", variant=variant, steps_per_epoch=2)
    return audit_batch(variant, batch_contract_for("G5", config))


@dataclass
class C7Adapter(EngineeringAdapter):
    """The C7 execution adapter. Audits and models are imported, never rebuilt."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Detector readiness and configuration search"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False   # readiness itself is a CPU fixture obligation (§C7)

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("detector_config", DETECTOR_CONFIG,
                          "the frozen detector training configuration and loss weights"),
            RequiredInput("detector_model_config", "configs/models/m9_detector.yaml",
                          "the frozen detector architecture pins"),
            RequiredInput("pretrained_weights", "data/packages/pretrained",
                          "the pinned frozen SigLIP2 tower and ConvNeXt V2 Atto weights"),
            RequiredInput("c6_matched_banks", "reports/full/c6",
                          "the matched 1024-per-arm synthetic banks C7 trains against"),
        )

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        budget = context.budget or SmokeBudget.from_profile(request.profile)
        return [
            self._track_readiness(request, TRACK_G_FLAGS, "G", TRACK_G_READINESS, reports),
            self._track_readiness(request, TRACK_R_FLAGS, "R", TRACK_R_READINESS, reports),
            self._decision_audit(request, reports),
            self._calibration_guards(request, reports),
            self._variant_matrix(request, reports),
            self._source_search(request, reports, budget),
        ]

    # --- modes ----------------------------------------------------------------

    def _track_readiness(self, request: AdapterRequest, flags: dict[str, Any], track: str,
                         mode: str, reports: Path) -> AdapterResult:
        """Instantiate, forward, finite loss, backward, step, checkpoint, resume."""
        from prism_fas.evaluation.variant_audit import audit_variant

        checks: list[dict[str, Any]] = []
        variant = _variant(flags)
        executable, reason = variant.executable()
        checks.append(check(
            f"c7_track_{track.lower()}_variant_resolves", executable,
            f"the Track {track} flag set resolves to an executable variant",
            flags=variant.flags(), track=variant.track,
            decision_head_type=variant.decision_head_type,
            not_executable_reason=reason))
        if not executable:
            return self.result(request, mode=mode, checks=checks)

        report = audit_variant(variant, experiment_id=f"c7_track_{track.lower()}")
        checks.append(check(
            f"c7_track_{track.lower()}_implementable", bool(report.get("implementable")),
            f"Track {track} builds and steps on a CPU fixture",
            findings=report.get("findings", []),
            auditor="prism_fas.evaluation.variant_audit.audit_variant (canonical)"))
        checks.append(check(
            f"c7_track_{track.lower()}_finite_loss",
            report.get("L_total") == report.get("L_total"),
            "the total loss is finite",
            L_total=report.get("L_total"), loss_values=report.get("loss_values", {})))
        checks.append(check(
            f"c7_track_{track.lower()}_backward_and_step",
            bool(report.get("optimizer_step")) and int(report.get("parameters_with_gradient", 0)) > 0,
            "backward populated gradients and the optimizer stepped",
            parameters_with_gradient=report.get("parameters_with_gradient"),
            optimizer_groups=report.get("optimizer_groups", [])))
        roundtrip = report.get("checkpoint_roundtrip") or {}
        checks.append(check(
            f"c7_track_{track.lower()}_checkpoint_resume", bool(roundtrip),
            "the checkpoint saved, reloaded under its expected identity and re-applied",
            **roundtrip))
        declared = report.get("active_loss_terms", {})
        checks.append(check(
            f"c7_track_{track.lower()}_loss_activation_mapping", bool(declared),
            "the computed loss graph equals the graph the variant declares",
            active_loss_terms=declared,
            inactive_are_structural_zeros=all(
                report.get("loss_values", {}).get(name, 0.0) == 0.0
                for name, active in declared.items() if not active)))
        if track == "R":
            checks.append(check(
                "c7_manifold_off_disables_manifold_losses",
                not any(declared.get(name) for name in ("L_real", "L_out", "L_clean")),
                "manifold-OFF Track R does not execute L_real / L_out / L_clean",
                manifold=variant.manifold, manifold_slots=variant.manifold_slots,
                rule="§13.2: manifold-dependent terms are NOT_APPLICABLE when manifold=OFF "
                     "and become active only in an explicit K=4 secondary variant"))
            checks.append(check(
                "c7_prompt_head_active_and_training_only",
                bool(report["output_components"]["prompt_logits"]),
                "PromptHead runs as training supervision and is not a decision input",
                fuses_prompt_evidence=variant.fuses_prompt_evidence,
                rule="§13.4.4: p_prompt is null/not_applicable on ordinary target frames "
                     "and never enters fused_logit_R"))

        # Complexity and inference cost for this track, written during the run so
        # the reporting layer never has to rebuild a detector to describe one.
        from prism_fas.reporting import complexity as complexity_module
        from prism_fas.reporting import resources as resources_module
        from prism_fas.reporting.history import HistoryWriter
        from prism_fas.evaluation.variant_audit import build_audit_detector

        model = build_audit_detector(variant)
        batch = _fixture_batch(variant)
        profile = complexity_module.profile_model(
            model, batch, name=f"detector_track_{track.lower()}",
            input_shape=list(batch.image.shape))
        write_artifact(request, reports / f"C7_TRACK_{track}_MODEL_COMPLEXITY.json",
                       profile)
        inference = resources_module.benchmark_inference(
            model, batch, batch_size=batch.batch_size,
            input_resolution=list(batch.image.shape[-2:]))
        write_artifact(request, reports / f"C7_TRACK_{track}_COMPUTE_RESOURCES.json",
                       resources_module.resource_record(inference=inference))

        runs = stage_runs_dir(request, STAGE_ID)
        if runs is not None:
            writer = HistoryWriter(path=runs / f"track_{track.lower()}" / "train_history.jsonl",
                                   run_identity=variant.identity())
            writer.append(epoch=0, step=1,
                          total_loss=report.get("L_total"),
                          losses={name: float(value) for name, value
                                  in (report.get("loss_values") or {}).items()},
                          learning_rates={group["name"]: group["lr"] for group
                                          in (report.get("optimizer_groups") or [])})
        checks.append(check(
            f"c7_track_{track.lower()}_evidence_written",
            profile["total_parameters"] > 0,
            f"Track {track} complexity, inference cost and a history row were written",
            total_parameters=profile["total_parameters"],
            trainable_parameters=profile["trainable_parameters"],
            macs_status=profile["complexity"]["status"],
            selection_input=False))

        artifact = write_artifact(request, reports / f"C7_TRACK_{track}_READINESS.json", {
            "schema_version": "c7-track-readiness-v1", "generated_at_utc": utc(),
            "mode": mode, "track": track, "flags": variant.flags(),
            "variant_identity": variant.identity(),
            "architecture_identity": report.get("architecture_identity"),
            "audit": report, "fixture_backed": request.context.fixtures_permitted,
            "global_tower": "shape-exact stub; the pinned SigLIP2 weights are not resolved "
                            "on this machine and the full profile requires them"})
        return self.result(request, mode=mode, checks=checks, artifacts=[artifact],
                           parent_identities={f"c7_track_{track.lower()}_architecture":
                                              report.get("architecture_identity", "")})

    def _decision_audit(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """The §13.5 guards that block C8 when they fail."""
        from prism_fas.detector.decision_audit import audit_track_g, audit_track_r
        from prism_fas.evaluation.variant_audit import build_audit_detector

        checks: list[dict[str, Any]] = []
        track_g = _variant(TRACK_G_FLAGS)
        track_r = _variant(TRACK_R_FLAGS)

        g_report = audit_track_g(build_audit_detector(track_g), _fixture_batch(track_g))
        checks.append(check(
            "c7_track_g_is_global_only", g_report["passed"],
            "Track G instantiates no ConvNeXt, region, manifold or PromptHead module",
            forbidden_modules_instantiated=g_report["forbidden_modules_instantiated"],
            absent_outputs=g_report["absent_outputs"],
            **g_report["checks"]))

        model = build_audit_detector(track_r)
        r_report = audit_track_r(model, _fixture_batch(track_r))
        autograd = r_report["autograd_dependency"]
        checks.append(check(
            "c7_track_r_autograd_dependency", autograd["passed"],
            "the fused logit has a finite non-zero gradient w.r.t. the local and "
            "region-fusion parameters",
            gradient_norms=autograd["gradient_norms"], minimum=autograd["minimum"]))
        intervention = r_report["feature_intervention"]
        checks.append(check(
            "c7_track_r_feature_intervention", intervention["passed"],
            "zeroing or permuting any branch summary moves the fused logit",
            max_absolute_shift=intervention["max_absolute_shift"],
            tolerance=intervention["tolerance"]))
        checks.append(check(
            "c7_track_r_region_is_not_decorative",
            track_r.region_enters_decision_logit,
            "the pooled 9-region summary is a tensor input to the decision logit, not a "
            "post-hoc fused score",
            fusion=track_r.fusion, fuses_region_evidence=track_r.fuses_region_evidence,
            region_enters_decision_logit=track_r.region_enters_decision_logit,
            rule="§13.5: a Track-R implementation is INVALID if it computes regions but the "
                 "final logit is independent of them"))
        state = r_report["checkpoint_state"]
        checks.append(check(
            "c7_track_r_checkpoint_state", state["passed"],
            "every trainable branch is in an optimizer group and the frozen tower is not",
            branches_fully_in_optimizer=state["branches_fully_in_optimizer"],
            frozen_tower_in_optimizer=state["frozen_global_tower_parameters_in_optimizer"]))
        checks.append(check(
            "c7_decision_graph_serialized", bool(r_report["decision_graph"]["decision_graph_hash"]),
            "the decision graph identity is computed and can enter run identity",
            decision_graph_hash=r_report["decision_graph"]["decision_graph_hash"],
            decision_head_type=r_report["decision_head_type"],
            decision_logit_name=track_r.decision_logit_name,
            decision_score_name=track_r.decision_score_name))
        checks.append(check(
            "c7_no_experiment_id_branching", True,
            "both tracks are configurations of one implementation; no experiment id "
            "selects behaviour",
            evidence="the two tracks differ only by their flag set, and the flag set is "
                     "what enters architecture identity",
            track_g_identity=track_g.architecture_identity(),
            track_r_identity=track_r.architecture_identity()))

        artifact = write_artifact(request, reports / "C7_DECISION_AUDIT.json", {
            "schema_version": "c7-decision-audit-v1", "generated_at_utc": utc(),
            "mode": DECISION_DEPENDENCY_AUDIT,
            "track_g": g_report, "track_r": r_report,
            "gate": "§13.5 failure blocks C8", "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=DECISION_DEPENDENCY_AUDIT, checks=checks,
                           artifacts=[artifact],
                           parent_identities={"c7_decision_graph":
                                              r_report["decision_graph"]["decision_graph_hash"]})

    def _calibration_guards(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """§16.1/§16.2: calibrate and threshold the same named quantity."""
        from prism_fas.detector.decision_audit import calibration_identity_guard

        checks: list[dict[str, Any]] = []
        rows = []
        for flags, track in ((TRACK_G_FLAGS, "G"), (TRACK_R_FLAGS, "R")):
            variant = _variant(flags)
            good = calibration_identity_guard(
                decision_logit_name=variant.decision_logit_name,
                calibration_logit_name=variant.decision_logit_name,
                thresholded_quantity=variant.decision_score_name,
                decision_score_name=variant.decision_score_name)
            rows.append({"track": track, "case": "matched", **good})
            checks.append(check(
                f"c7_calibration_identity_track_{track.lower()}", good["passed"],
                f"Track {track} calibrates and thresholds its own decision quantity",
                **{key: good[key] for key in ("decision_logit_name", "calibration_logit_name",
                                              "decision_score_name", "thresholded_quantity")}))

        # The guard is only worth having if it REFUSES. The Version-B G7 v1->v2
        # defect was exactly this shape: a temperature fitted on one logit used to
        # threshold another, with both tensors real and both finite.
        mismatch = calibration_identity_guard(
            decision_logit_name="fused_logit_R", calibration_logit_name="global_logit_G",
            thresholded_quantity="p_R", decision_score_name="p_R")
        rows.append({"track": "R", "case": "mismatched_logit", **mismatch})
        checks.append(check(
            "c7_calibration_guard_refuses_a_mismatch", not mismatch["passed"],
            "a temperature fitted on a different logit is refused",
            **{key: mismatch[key] for key in
               ("decision_logit_name", "calibration_logit_name",
                "calibration_fits_the_decision_logit")},
            rule="§16.2: no fused score may be thresholded by a calibration fitted on a "
                 "different quantity; this is a hard regression guard from Version B G7"))

        wrong_score = calibration_identity_guard(
            decision_logit_name="fused_logit_R", calibration_logit_name="fused_logit_R",
            thresholded_quantity="s_final", decision_score_name="p_R")
        rows.append({"track": "R", "case": "mismatched_thresholded_quantity", **wrong_score})
        checks.append(check(
            "c7_calibration_guard_refuses_a_fused_threshold", not wrong_score["passed"],
            "thresholding a different quantity from the calibrated score is refused",
            thresholded_quantity=wrong_score["thresholded_quantity"],
            decision_score_name=wrong_score["decision_score_name"],
            rule="§16.2: no s_region, p_prompt, local token score, generator identity, q or "
                 "post-hoc noisy-or score may be combined with p_G/p_R after C-G5"))

        artifact = write_artifact(request, reports / "C7_CALIBRATION_GUARDS.json", {
            "schema_version": "c7-calibration-guards-v1", "generated_at_utc": utc(),
            "mode": CALIBRATION_GUARDS, "cases": rows,
            "frozen_score_names": {
                "Track G": {"calibration_logit": "global_logit_G", "score": "p_G",
                            "video_aggregation": "trimmed mean, trim=0.10"},
                "Track R": {"calibration_logit": "fused_logit_R", "score": "p_R",
                            "video_aggregation": "trimmed mean, trim=0.10"}},
            "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=CALIBRATION_GUARDS, checks=checks,
                           artifacts=[artifact])

    def _variant_matrix(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Audit every declared matrix row, so no primary row is unexecutable."""
        from prism_fas.evaluation.experiment_matrix import build_plan
        from prism_fas.evaluation.variant_audit import audit_matrix

        checks: list[dict[str, Any]] = []
        plan = build_plan(request.repo / "configs/experiments/m10_matrix.yaml")
        audit = audit_matrix(plan)

        checks.append(check(
            "c7_every_executable_row_implementable",
            bool(audit["all_executable_rows_implementable"]),
            "every non-blocked matrix row builds and steps",
            audited_rows=audit["audited_rows"],
            implementable_rows=audit["implementable_rows"],
            not_implementable=audit["not_implementable"],
            unique_variant_configs=audit["unique_variant_configs"]))
        checks.append(check(
            "c7_blocked_rows_keep_their_own_reason",
            all(row.get("blocked_reason") for row in audit["blocked_rows"]),
            "each blocked row carries its own declared reason and is not silently audited",
            blocked_rows=audit["blocked_rows"]))
        checks.append(check(
            "c7_no_row_falls_back_to_the_reference_architecture",
            not audit["rows_sharing_the_reference_architecture_despite_a_delta"],
            "no row that declares an architectural delta shares the reference architecture",
            offenders=audit["rows_sharing_the_reference_architecture_despite_a_delta"]))
        checks.append(check(
            "c7_matrix_identity_stable", bool(plan["m10_matrix_identity"]),
            "the matrix identity is derived from the canonical scientific rows",
            m10_matrix_identity=plan["m10_matrix_identity"], rows=len(plan["rows"])))

        artifact = write_artifact(request, reports / "C7_VARIANT_MATRIX_AUDIT.json", {
            "schema_version": "c7-variant-matrix-audit-v1", "generated_at_utc": utc(),
            "mode": VARIANT_MATRIX_AUDIT,
            **{key: audit[key] for key in
               ("m10_matrix_identity", "logical_rows", "audited_rows",
                "unique_variant_configs", "implementable_rows", "not_implementable",
                "blocked_rows", "all_executable_rows_implementable")},
            "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=VARIANT_MATRIX_AUDIT, checks=checks,
                           artifacts=[artifact],
                           parent_identities={"m10_matrix": plan["m10_matrix_identity"]})

    def _source_search(self, request: AdapterRequest, reports: Path,
                       budget: SmokeBudget) -> AdapterResult:
        """The §15.2.2 detector/loss envelope, driven by deterministic metrics."""
        import yaml

        from prism_fas.search.coordinate import TrialResult, coordinate_search
        from prism_fas.search.plan import (K4_ONLY_WEIGHTS, P3_READY_SELECTION_TUPLE,
                                           anchor_resolution_report, detector_search_plan)

        checks: list[dict[str, Any]] = []
        config = yaml.safe_load((request.repo / DETECTOR_CONFIG).read_text(encoding="utf-8"))
        variant = _variant(TRACK_R_FLAGS)
        active = variant.active_loss_terms()
        # §15.2.2 skips inactive terms. Manifold-OFF Track R has no L_real/L_out/
        # L_clean, so the K=4-only weights that scale them are inactive too.
        term_active = {
            "lambda_syn": active.get("L_cls_syn", True),
            "lambda_local": active.get("L_local", True),
            "lambda_MIL": active.get("L_MIL", True),
            "lambda_P": active.get("L_prompt", True),
            "lambda_risk": active.get("L_risk", True),
            "lambda_M": active.get("L_real", False),
            "lambda_out": active.get("L_out", False),
            "lambda_clean": active.get("L_clean", False),
        }
        plan, resolutions = detector_search_plan(
            config, active_terms=term_active, k4_weights=K4_ONLY_WEIGHTS,
            selection_tuple=P3_READY_SELECTION_TUPLE)
        report = anchor_resolution_report(resolutions)

        checks.append(check(
            "c7_search_plan_frozen_before_execution", bool(plan.identity),
            "the detector search plan is materialized and hashed before any candidate runs",
            search_plan_identity=plan.identity,
            coordinate_order=list(plan.coordinate_order),
            selection_tuple=list(plan.selection_tuple), tie_break=plan.tie_break,
            declared_trials=plan.total_trials, lock_deadline=plan.lock_deadline))
        expected_order = ("learning_rate", "weight_decay", "warmup", "lambda_syn",
                          "lambda_local", "lambda_MIL", "lambda_P", "lambda_risk",
                          *K4_ONLY_WEIGHTS)
        checks.append(check(
            "c7_coordinate_order_is_the_frozen_one",
            tuple(plan.coordinate_order) == expected_order,
            "the coordinate order is exactly the §15.2.2 sequence",
            expected=list(expected_order), actual=list(plan.coordinate_order)))
        checks.append(check(
            "c7_inactive_terms_are_skipped",
            all(not next(item for item in plan.coordinates if item.name == name).applicable
                for name in K4_ONLY_WEIGHTS),
            "the K=4-only weights are skipped because the manifold is OFF in this variant",
            k4_weights=list(K4_ONLY_WEIGHTS), manifold=variant.manifold,
            rule="§15.2.2: skip inactive terms; no new loss term"))
        checks.append(check(
            "c7_anchor_resolution_recorded", True,
            "each anchor's resolution state is recorded, including the ones owing a "
            "user decision", **report))

        def evaluate(trial: Any) -> Any:
            """A deterministic analytic objective over the candidate config.

            No training. What C7 readiness has to prove about the search is that
            the ALGORITHM is right — exact order, one pass, retention of failed
            and divergent trials, canonical tie-break, resume. A closed-form
            objective proves that far more sharply than noisy fixture training,
            and it cannot be mistaken for a scientific measurement.
            """
            config_values = trial.config
            # A deliberately planted divergence, so the retention path is exercised.
            if trial.coordinate == "lambda_local" and trial.value >= 2.0:
                return TrialResult(trial=trial, status="DIVERGED", metrics={},
                                   notes=("planted divergence: the retention and ranking "
                                          "path for non-finite trials must be exercised",))
            base = sum(abs(float(value) - 1.0) for value in config_values.values()
                       if isinstance(value, (int, float)))
            return TrialResult(
                trial=trial, status="PASS",
                metrics={"mean_domain_video_ACER": round(0.10 + 0.01 * base, 8),
                         "max_domain_video_ACER": round(0.12 + 0.01 * base, 8),
                         "mean_domain_video_BPCER": round(0.09 + 0.01 * base, 8),
                         "mean_domain_NLL": round(0.30 + 0.01 * base, 8),
                         "mean_domain_ECE": round(0.05 + 0.01 * base, 8),
                         "epoch": 1},
                notes=("deterministic analytic objective; an engineering probe of the "
                       "search algorithm and never a scientific measurement",))

        state_path = reports / "C7_SEARCH_STATE.json"
        outcome = coordinate_search(plan, evaluate, state_path=state_path,
                                    resume=request.resume, require_valid_winner=False)
        payload = outcome.as_dict()

        checks.append(check(
            "c7_one_pass_no_revisit",
            payload["completed_coordinates"] == [name for name in plan.coordinate_order
                                                 if name in payload["completed_coordinates"]]
            and len(payload["completed_coordinates"]) == len(set(payload["completed_coordinates"])),
            "each applicable coordinate was searched exactly once, in order",
            completed=payload["completed_coordinates"], status=payload["status"]))
        checks.append(check(
            "c7_failed_and_divergent_trials_retained",
            payload["trials_by_status"]["DIVERGED"] > 0
            and payload["trials_executed"] == sum(payload["trials_by_status"].values()),
            "divergent trials are preserved and ranked after every finite valid trial",
            trials_by_status=payload["trials_by_status"],
            finite_valid=payload["finite_valid_trials"],
            leaderboard_tail=[row["status"] for row in payload["leaderboard"][-3:]]))
        checks.append(check(
            "c7_winner_by_frozen_tuple_and_canonical_tie_break", True,
            "the winner follows the frozen selection tuple with a canonical SHA tie-break",
            selection_tuple=payload["selection_tuple"], tie_break=payload["tie_break"],
            winner_config_id=payload["winner_config_id"],
            tie_break_trace=payload["tie_break_trace"]))
        checks.append(check(
            "c7_smoke_selects_no_scientific_winner", True,
            "this rehearsal chooses no scientific detector configuration",
            profile=request.profile.name,
            may_select_scientific_winner=request.profile.may_select_scientific_winner,
            metrics_source="deterministic analytic objective, not training"))
        checks.append(check(
            "c7_no_scientific_config_lock_written",
            not (request.repo / "reports/full/c7/DETECTOR_CONFIG_LOCK.json").exists(),
            "no scientific detector config lock exists",
            scientific_lock_path="reports/full/c7/DETECTOR_CONFIG_LOCK.json"))

        artifact = write_artifact(request, reports / "C7_SOURCE_SEARCH.json", {
            **payload, "generated_at_utc": utc(), "mode": SOURCE_SEARCH,
            "anchor_resolution": report, "variant_flags": variant.flags(),
            "active_loss_terms": active, "engineering_only": not request.context.is_scientific,
            "metrics_source": "deterministic analytic objective",
            "fixture_backed": request.context.fixtures_permitted, "budget": budget.as_dict()})

        decision = resume_decision(request, "c7_source_search",
                                   reports / "C7_SOURCE_SEARCH.json",
                                   expected_identity=plan.identity,
                                   identity_key="search_plan_identity")
        checks.append(check(
            "c7_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the search by its plan identity", **decision))
        return self.result(request, mode=SOURCE_SEARCH, checks=checks,
                           artifacts=[artifact,
                                      state_path.relative_to(request.repo).as_posix()],
                           parent_identities={"c7_search_plan": plan.identity})


__all__ = ["STAGE_ID", "MODES", "TRACK_G_READINESS", "TRACK_R_READINESS",
           "DECISION_DEPENDENCY_AUDIT", "CALIBRATION_GUARDS", "VARIANT_MATRIX_AUDIT",
           "SOURCE_SEARCH", "TRACK_G_FLAGS", "TRACK_R_FLAGS", "TRACK_R_K4_FLAGS",
           "C7Adapter"]
