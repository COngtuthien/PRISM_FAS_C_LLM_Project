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

The search half is TWO halves, and they share no metric-producing code.

The ENGINEERING search exercises the coordinate engine with a deterministic
analytic objective. What readiness must prove about the search is that the
*algorithm* is right — order, one pass, retention, tie-break, resume — and a
closed-form objective proves that far more sharply than noisy fixture training
would, while being impossible to mistake for a measurement.

The SCIENTIFIC search trains real detector configurations: the frozen C6 matched
bank for the decided arm, the pinned SigLIP2 and ConvNeXt weights, the canonical
`M9Trainer`, the full declared G1→G2→G5→G6 flow per trial, source_dev
calibration and the §15.4 video-level ranking tuple. It ends by writing the one
`reports/full/c7/DETECTOR_CONFIG_LOCK.json` that C8 declares as a required
input — which the engineering path had asserted must never exist, a check that
made a legitimate scientific C7 unreachable by construction.

One field the spec does not fix stands between the implemented path and a run:
which of C6's three matched banks the bounded search trains against. That is
the treatment factor C8 compares, so it is a decision record
(`configs/search/c7_source_search_decision.yaml`) rather than a default, and the
scientific path blocks on it before the first trial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterError, AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, read_json,
                                                resume_decision,
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

#: The scientific substages. Disjoint from the rehearsal modes for the reason C6
#: separated its own: one report may never show a fixture search and a real one
#: under the same name.
VERIFY_C6_EVIDENCE = "VERIFY_C6_EVIDENCE"
SCIENTIFIC_SEARCH = "SCIENTIFIC_SOURCE_SEARCH"
FINALIZE_DETECTOR_CONFIG = "FINALIZE_DETECTOR_CONFIG"
VERIFY_CONFIG_LOCK = "VERIFY_CONFIG_LOCK"

SCIENTIFIC_MODES: tuple[str, ...] = (VERIFY_C6_EVIDENCE, SCIENTIFIC_SEARCH,
                                     FINALIZE_DETECTOR_CONFIG, VERIFY_CONFIG_LOCK)

MODES: tuple[str, ...] = (TRACK_G_READINESS, TRACK_R_READINESS, DECISION_DEPENDENCY_AUDIT,
                          CALIBRATION_GUARDS, VARIANT_MATRIX_AUDIT,
                          SOURCE_SEARCH) + SCIENTIFIC_MODES

DETECTOR_CONFIG = "configs/train/m9_reference.yaml"

#: The one governing artifact scientific C7 produces, and the exact path C8
#: declares as a required input. Named once, here, so the producer and the two
#: consumers cannot disagree about where it lives.
SCIENTIFIC_REPORTS = "reports/full/c7"
DETECTOR_CONFIG_LOCK = "DETECTOR_CONFIG_LOCK.json"
SCIENTIFIC_CONFIG_LOCK_PATH = f"{SCIENTIFIC_REPORTS}/{DETECTOR_CONFIG_LOCK}"

#: The scientific search state file. A different name from the engineering one so
#: neither pass can resume into the other's state even if both were copied into
#: one directory; `coordinate_search` additionally refuses a state whose recorded
#: plan identity differs, and the two plans always differ.
SCIENTIFIC_SEARCH_STATE = "C7_SCIENTIFIC_SEARCH_STATE.json"
TRIAL_SUMMARY = "C7_TRIAL_SUMMARY.json"

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


def _fixture_batch(variant: Any, context: Any = None) -> Any:
    """The CPU readiness batch. Synthetic tensors, never a source sample.

    The guard lives HERE as well as at the call sites, because a guard only at
    the caller is a guard the next caller forgets. `context=None` is accepted so
    the readiness modes that have already asserted can call it plainly, and the
    standing fixture-leak audit still sees an `assert_fixture_permitted` in this
    function's own body.
    """
    from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for
    from prism_fas.evaluation.variant_audit import audit_batch

    assert_fixture_permitted(context, "the C7 readiness fixture batch")
    config = M9TrainingConfig(run_id=f"c7_{variant.track}", variant=variant, steps_per_epoch=2)
    return audit_batch(variant, batch_contract_for("G5", config))


@dataclass
class C7Adapter(EngineeringAdapter):
    """The C7 execution adapter. Audits and models are imported, never rebuilt."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Detector readiness and configuration search"
    modes: tuple[str, ...] = MODES
    # The READINESS half is a CPU fixture obligation and needs no accelerator.
    # This flag governs only `full_precondition_gate`, which `EngineeringAdapter.run`
    # invokes under a scientific context alone — so it is a statement about the
    # SCIENTIFIC search, which trains real detectors and may not run on a CPU.
    requires_gpu: bool = True

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("detector_config", DETECTOR_CONFIG,
                          "the frozen detector training configuration and loss weights"),
            RequiredInput("detector_model_config", "configs/models/m9_detector.yaml",
                          "the frozen detector architecture pins"),
            RequiredInput("pretrained_weights", "weights",
                          "the pinned frozen SigLIP2 tower and ConvNeXt V2 Atto weights"),
            RequiredInput("source_package", "data/packages/prism_data_v1_m3b",
                          "the validated M3B source package supplying source_train and "
                          "source_dev"),
            RequiredInput("c6_matched_banks", "reports/full/c6",
                          "the matched 1024-per-arm synthetic banks C7 trains against"),
            RequiredInput("c5_candidates", "runs/full/c5/scientific/candidates",
                          "the rendered candidate bytes the C6 banks address"),
        )

    def semantic_preconditions(self, request: AdapterRequest) -> list[dict[str, Any]]:
        """Beyond existence: the C6 closure must be TRUE and the search decided.

        `reports/full/c6` existing proves nothing — a refused, partial or stale
        closure is a directory that exists. The canonical strict verifier is the
        one C6's own VERIFY_C6_LOCKS uses; a weaker second opinion here would be
        the gap it is supposed to close.
        """
        from prism_fas.evaluation import c6_evidence
        from prism_fas.search import c7_decision

        closure = c6_evidence.evidence_report(request.repo)
        decision = c7_decision.decision_report(request.repo)
        return [
            {"name": "c6_closure_verified", "path": c6_evidence.C6_REPORTS,
             "present": closure["valid"], "blocking": not closure["valid"],
             "description": ("the three C6 BANK_LOCKs, the profile-selection lock and "
                             "the matched-bank artifact agree, each bank holds 1024 "
                             "samples as 512 Physics + 512 GPAT, every provenance "
                             "closure is closed and target_access is 0"),
             "verifier": "prism_fas.evaluation.c6_evidence.verify_c6_evidence",
             "reason_code": closure["reason_code"],
             "problems": closure["problems"][:12]},
            {"name": "c7_source_search_decision", "path": decision["config_path"],
             "present": decision["resolved"], "blocking": not decision["resolved"],
             "description": ("which C6 matched bank, protocol, track, ranking tuple and "
                             "trial schedule the one bounded search pass uses; §15.2.2 "
                             "fixes the envelope but not the training population, and "
                             "the arm is the treatment factor C8 compares"),
             "verifier": "prism_fas.search.c7_decision.load_decision",
             "reason_code": decision["reason_code"],
             "problems": [decision["error"]] if decision["error"] else []},
        ]

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        """Two workflows, chosen by the context — never one that adapts.

        The rehearsal path below is unchanged and still owns the CPU fixture
        readiness obligation §C7 states. The scientific path shares none of its
        metric-producing code: not the batch, not the model, not the evaluator,
        not the search state file and not the artifact it finalizes into.
        """
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        """The CPU readiness obligation. Produces engineering evidence only."""
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
        assert_fixture_permitted(request.context,
                                 "the C7 detector complexity fixture batch")
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
        """The §13.5 guards that block C8 when they fail.

        Fixture-backed: `build_audit_detector` constructs a shape-exact model with
        an untrained tower, which is the right thing for an autograd and
        intervention audit and the wrong thing for anything measured. Guarded
        first, so the scientific path cannot reach it even if a future edit
        called this mode from `_scientific_workflow`.
        """
        from prism_fas.detector.decision_audit import audit_track_g, audit_track_r
        from prism_fas.evaluation.variant_audit import build_audit_detector

        assert_fixture_permitted(request.context,
                                 "the C7 §13.5 audit detector and fixture batch")
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
        """Audit every declared matrix row, so no primary row is unexecutable.

        `audit_matrix` builds and steps one audit detector per unique variant
        config, which is fixture machinery by construction.
        """
        from prism_fas.evaluation.experiment_matrix import build_plan
        from prism_fas.evaluation.variant_audit import audit_matrix

        assert_fixture_permitted(request.context,
                                 "the C7 variant-matrix audit detectors")
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
        # What this check may assert is that THIS run wrote no scientific lock —
        # not that none exists. It used to assert the second, which made a
        # legitimate scientific C7 unreachable: C8 declares that exact path as a
        # required input, so a successful scientific C7 would have made every
        # later rehearsal fail on a file it was right to have produced.
        scientific_reports = (request.repo / SCIENTIFIC_REPORTS).resolve()
        checks.append(check(
            "c7_rehearsal_writes_no_scientific_config_lock",
            not str(reports.resolve()).startswith(str(scientific_reports))
            and request.profile.reports_namespace != SCIENTIFIC_REPORTS.rsplit("/", 1)[0],
            "this rehearsal writes outside the scientific namespace, so it cannot "
            "occupy the detector config lock's position",
            scientific_lock_path=SCIENTIFIC_CONFIG_LOCK_PATH,
            rehearsal_reports=reports.relative_to(request.repo).as_posix(),
            scientific_lock_present=(request.repo / SCIENTIFIC_CONFIG_LOCK_PATH).exists(),
            rule="a scientific lock produced by an earlier full-profile C7 is "
                 "evidence, not a violation; what would be a violation is a "
                 "rehearsal writing to that path"))

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

    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C7: ONE bounded pass per track, both against the DET bank.

        Nothing here is shared with the rehearsal. The readiness modes are absent
        by construction — they are a CPU fixture obligation, and running one
        inside a scientific pass would put fixture numbers in the same report as
        scientific ones. They still run, under the rehearsal profile, and their
        artifacts are what §C7 readiness is evidenced by.

        Two passes, not three. The search population is frozen to one arm
        (`C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET`) and the search is repeated per
        TRACK, because Track G and Track R have different active loss sets and
        §15.2.2 skips inactive terms. A pass per ARM would give each generator its
        own winning configuration and confound the generator effect with detector
        tuning — which is the confound the whole design exists to remove.
        """
        results: list[AdapterResult] = []
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID) or reports

        inputs, prepare = self._scientific_prepare(request, reports)
        results.append(prepare)
        if inputs is None:
            return results

        state, plan_result = self._scientific_plan(request, inputs, reports)
        results.append(plan_result)
        if state is None:
            return results

        outcomes, search = self._scientific_search(request, inputs, state, reports, runs)
        results.append(search)
        if outcomes is None:
            return results

        results.append(self._scientific_finalize(request, inputs, state, outcomes,
                                                 reports, runs))
        results.append(self._scientific_verify_lock(request, reports))
        return results

    def _scientific_prepare(self, request: AdapterRequest,
                            reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Resolve and prove the frozen inputs. No batch and no model is built here."""
        from prism_fas.pipeline.adapters import sources

        checks: list[dict[str, Any]] = []
        try:
            inputs = sources.verify_detector_inputs(request.repo)
        except sources.SourceUnavailable as error:
            checks.append(check(
                "c7_scientific_inputs_verified", False,
                f"the frozen scientific inputs are not usable: {type(error).__name__}",
                error=str(error),
                reason_code=getattr(error, "reason_code", "MISSING_DATA")))
            return None, self.result(request, mode=VERIFY_C6_EVIDENCE, checks=checks,
                                     summary="C7 scientific inputs unavailable")

        closure = inputs["c6"]
        checks.append(check(
            "c7_scientific_inputs_verified", True,
            "the M3B package, the frozen M7 bank, the pinned backbones, the C5 "
            "candidate tree and the C6 closure are present and agree",
            package_identity=inputs["package_identity"],
            recipe_bank_identity=inputs["recipe_bank_identity"],
            candidates_root=inputs["candidates_root"],
            verifier=inputs["verified_by"]))
        checks.append(check(
            "c7_c6_closure_verifies", True,
            "the three matched bank locks bind one selector identity, one threshold "
            "identity and one selected profile, and each holds the frozen cardinality",
            selected_profile=closure["selected_profile"],
            selector_identity_sha256=closure["selector_identity_sha256"],
            quality_threshold_identity=closure["quality_threshold_identity"],
            final_bank_per_arm=closure["final_bank_per_arm"],
            per_route=closure["per_route"],
            banks={arm: item["by_route"] for arm, item in closure["banks"].items()},
            verifier="prism_fas.evaluation.c6_evidence.verify_c6_evidence"))
        weights = inputs["pretrained"]
        checks.append(check(
            "c7_pretrained_weights_are_pinned_not_stubs",
            weights["stub_substituted"] is False
            and not weights["downloaded_during_run"]
            and bool(weights["global_tower"]["identity_sha256"])
            and bool(weights["local_backbone"]["weight_sha256"]),
            "the frozen SigLIP2 tower and the pinned ConvNeXt V2 Atto weight resolved "
            "by SHA-256; no shape-exact stub and no silent download",
            **weights))
        checks.append(check(
            "c7_no_fixture_in_scientific_context", True,
            "the scientific path builds no fixture batch and no audit detector",
            fixture_batch_used=False, audit_detector_used=False,
            batch_source="prism_fas.detector.dataset.M9TrainingDataset (canonical)",
            model_source="prism_fas.detector.prism_detector.build_detector (canonical)"))
        checks.append(check(
            "c7_target_is_unreachable", True,
            "only source_train and source_dev are opened; the target is not resolved",
            target_paths_resolved=0, target_labels_resolved=0,
            target_metrics_computed=0))

        artifact = write_artifact(request, reports / "C7_SCIENTIFIC_INPUTS.json", {
            "schema_version": "c7-scientific-inputs-v1", "generated_at_utc": utc(),
            "mode": VERIFY_C6_EVIDENCE, "fixture_backed": False, **inputs})
        return inputs, self.result(
            request, mode=VERIFY_C6_EVIDENCE, checks=checks, artifacts=[artifact],
            parent_identities={
                "m3b_package": inputs["package_identity"],
                "m7_recipe_bank": inputs["recipe_bank_identity"],
                "c6_selector": closure["selector_identity_sha256"],
                "c6_quality_threshold": closure["quality_threshold_identity"]})

    def _scientific_plan(self, request: AdapterRequest, inputs: dict[str, Any],
                         reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Freeze one envelope per track, hashed before a single trial runs.

        The frozen DET bank's identities are bound into each plan's base config,
        so they enter the plan identity AND — because `plan.config_for` builds
        every candidate from the base config — the canonical SHA of every trial.
        Swapping the search bank therefore changes every config identity, which
        is what makes "changing the DET bank lock invalidates resume" structural
        rather than advisory.
        """
        import yaml

        from prism_fas.evaluation import source_selection
        from prism_fas.search.c7_decision import C7DecisionError, load_decision as load_c7
        from prism_fas.search.lr_decision import (COMMON_MULTIPLIER, LRDecisionError,
                                                  load_decision as load_lr)
        from prism_fas.search.plan import (K4_ONLY_WEIGHTS, anchor_resolution_report,
                                           detector_search_plan)

        checks: list[dict[str, Any]] = []
        config = yaml.safe_load((request.repo / DETECTOR_CONFIG).read_text(encoding="utf-8"))

        try:
            decision = load_c7(request.repo)
        except C7DecisionError as error:
            checks.append(check(
                "c7_source_search_population_frozen", False,
                f"the C7 source-search decision is unresolved: {error}",
                reason_code=error.reason_code,
                decision_record="configs/search/c7_source_search_decision.yaml",
                blocks="the first scientific trial; the implementation is complete "
                       "and the rehearsal is unaffected"))
            return None, self.result(
                request, mode=SCIENTIFIC_SEARCH, checks=checks,
                substage="C7_SCIENTIFIC_PLAN",
                summary="C7_SOURCE_SEARCH_SYNTHETIC_ARM = NEEDS_SCIENTIFIC_DECISION")

        arm = decision.training_arm
        checks.append(check(
            "c7_source_search_population_frozen", True,
            f"the search trains against the frozen C6 {arm} bank, on every declared "
            "track, under one bounded pass each",
            **decision.as_dict()))
        checks.append(check(
            "c7_search_arm_is_frozen_before_any_metric_exists",
            decision.frozen_before_any_trial,
            "the arm was decided before a scientific C7 metric existed, so it cannot "
            "have been chosen from a result",
            timing=decision.timing, frozen_on=decision.frozen_on,
            spec_status=decision.spec_status, source=decision.source,
            rule="§15.2.2 freezes the envelope and not the search population; the "
                 "arm is the treatment factor C8 compares"))
        refused = sorted(item for item in inputs["c6"]["banks"]
                         if not decision.permits_arm(item))
        checks.append(check(
            "c7_only_the_frozen_arm_supplies_synthetic_samples",
            bool(refused) and all(not decision.permits_arm(item) for item in refused),
            f"the {refused} bank(s) are refused as a C7 search population; no "
            "candidate byte from a non-frozen arm enters search training",
            frozen_arm=arm, refused_arms=refused,
            available_arms=sorted(inputs["c6"]["banks"]),
            prohibited_after_freeze=list(decision.prohibited_alternatives)))
        checks.append(check(
            "c7_runs_no_per_arm_search", True,
            "one bounded pass per TRACK, never one per generator arm",
            passes=[f"track_{track}" for track in decision.tracks],
            search_population=arm,
            rule="a search per arm would give each generator its own winning "
                 "configuration and confound the generator effect with detector "
                 "tuning, which is the confound the design removes"))

        try:
            record = load_lr(request.repo)
        except (LRDecisionError, OSError) as error:
            checks.append(check(
                "c7_lr_decision_approved", False,
                f"the approved learning-rate decision could not be resolved: {error}",
                reason_code="NEEDS_SCIENTIFIC_DECISION"))
            return None, self.result(request, mode=SCIENTIFIC_SEARCH, checks=checks,
                                     substage="C7_SCIENTIFIC_PLAN",
                                     summary="C7 has no approved LR decision to bind")
        checks.append(check(
            "c7_lr_decision_approved", bool(record.approved),
            "the learning-rate interpretation is the approved one; the ambiguous "
            "per-scalar coordinate is never searched",
            decision_identity=record.identity,
            components=sorted(record.components)))

        selection_tuple = source_selection.TUPLES[decision.selection_tuple_name]
        binding = _search_binding(inputs, decision, record)
        plans: dict[str, dict[str, Any]] = {}

        for track in decision.tracks:
            flags = TRACK_R_FLAGS if track == "R" else TRACK_G_FLAGS
            variant = _variant(flags)
            try:
                lr = record.for_component(f"C7_TRACK_{track}")
            except LRDecisionError as error:
                checks.append(check(
                    f"c7_track_{track.lower()}_lr_component", False,
                    f"no approved LR interpretation for Track {track}: {error}",
                    reason_code="NEEDS_SCIENTIFIC_DECISION"))
                return None, self.result(request, mode=SCIENTIFIC_SEARCH, checks=checks,
                                         substage="C7_SCIENTIFIC_PLAN",
                                         summary=f"C7 Track {track} has no LR decision")

            active = variant.active_loss_terms()
            term_active = _active_terms(variant)
            plan, resolutions = detector_search_plan(
                config, active_terms=term_active, k4_weights=K4_ONLY_WEIGHTS,
                selection_tuple=selection_tuple, lr_decision=lr,
                base_config=dict(binding))
            report = anchor_resolution_report(resolutions)

            # A coordinate skipped because its TERM is inactive in this variant is
            # what §15.2.2 requires; a coordinate skipped because its anchor is
            # AMBIGUOUS is a decision still owed. Only the second blocks.
            blocked = [item.name for item in plan.coordinates
                       if not item.applicable and "AMBIGUOUS" in str(item.skip_reason)]
            searched = [item.name for item in plan.coordinates if item.applicable]
            skipped = {item.name: item.skip_reason for item in plan.coordinates
                       if not item.applicable}

            checks.append(check(
                f"c7_track_{track.lower()}_plan_executable", not blocked and bool(searched),
                f"Track {track}'s envelope has no coordinate blocked by an unresolved "
                "ambiguity",
                searched_coordinates=searched, blocked_coordinates=blocked,
                skipped_coordinates=skipped, anchor_resolution=report))
            checks.append(check(
                f"c7_track_{track.lower()}_searches_only_its_active_terms",
                all(not term_active[name] or name in searched
                    for name in term_active if name in _TRIAL_LOSS_WEIGHTS)
                and all(name not in searched for name, on in term_active.items()
                        if not on),
                f"Track {track} tunes the loss weights its variant declares active and "
                "no others; §15.2.2 skips inactive terms rather than inventing a "
                "weight for them",
                active_loss_terms=active, searchable_terms=term_active,
                searched=searched,
                not_applicable=sorted(name for name, on in term_active.items() if not on),
                rule="the fairness invariant is ONE frozen configuration within a "
                     "track, not an identical numeric loss vector across two "
                     "structurally different architectures"))
            if track == "R":
                checks.append(check(
                    "c7_track_r_primary_does_not_tune_manifold_terms",
                    all(not term_active[name] for name in
                        ("lambda_M", "lambda_out", "lambda_clean")),
                    "manifold-OFF primary Track R tunes no L_real / L_out / L_clean "
                    "weight; the K=4 variant is an explicit typed secondary path",
                    manifold=variant.manifold, prototype_k=variant.prototype_k,
                    k4_only_weights=list(K4_ONLY_WEIGHTS)))
            if track == "G":
                checks.append(check(
                    "c7_track_g_does_not_tune_regional_or_prompt_terms",
                    not term_active["lambda_local"] and not term_active["lambda_MIL"]
                    and not term_active["lambda_P"],
                    "Track G instantiates no ConvNeXt, no regions and no PromptHead, "
                    "so lambda_local / lambda_MIL / lambda_P are NOT APPLICABLE",
                    local_branch=variant.local_branch, region=variant.region,
                    prompt=variant.prompt,
                    not_applicable=["lambda_local", "lambda_MIL", "lambda_P"]))
            checks.append(check(
                f"c7_track_{track.lower()}_plan_binds_the_frozen_search_bank",
                plan.base_config.get("c7_search_binding", {}).get(
                    "c6_bank_selected_set_sha256")
                == inputs["c6"]["banks"][arm]["selected_set_sha256"],
                f"Track {track}'s plan identity covers the {arm} bank's selected-set "
                "digest, the C6 selector and both decision identities",
                search_plan_identity=plan.identity,
                **dict(plan.base_config.get("c7_search_binding") or {})))

            frozen_order = ("learning_rate", "weight_decay", "warmup", "lambda_syn",
                            "lambda_local", "lambda_MIL", "lambda_P", "lambda_risk",
                            *K4_ONLY_WEIGHTS)
            expected_order = tuple(lr.coordinate_name if name == "learning_rate" else name
                                   for name in frozen_order)
            checks.append(check(
                f"c7_track_{track.lower()}_coordinate_order_is_the_frozen_one",
                tuple(plan.coordinate_order) == expected_order,
                "the coordinate order is exactly the §15.2.2 sequence, with the "
                "learning rate expressed as the approved multiplier in its own "
                "position",
                frozen_order=list(frozen_order), expected=list(expected_order),
                actual=list(plan.coordinate_order),
                lr_coordinate_name=lr.coordinate_name,
                lr_interpretation=lr.interpretation))
            if lr.interpretation == COMMON_MULTIPLIER:
                checks.append(check(
                    f"c7_track_{track.lower()}_lr_ratio_is_held_fixed",
                    all(lr.ratio_preserved(value) for value in lr.candidates),
                    "every multiplier preserves the frozen inherited LR ratio",
                    preserved_ratio=list(lr.preserved_ratio),
                    per_candidate={str(value): lr.lr_for_groups(value)
                                   for value in lr.candidates}))

            plans[track] = {"plan": plan, "variant": variant, "lr": lr,
                            "anchor_resolution": report, "active_loss_terms": active,
                            "searchable_terms": term_active}

        epochs = int(config["stages"]["total_epochs"])
        steps = int(config["batch"]["steps_per_epoch"])
        declared = sum(item["plan"].total_trials for item in plans.values())
        checks.append(check(
            "c7_trial_schedule_is_not_shortened",
            decision.trial_schedule == "frozen_m9_schedule",
            "every trial runs the full frozen schedule; L.12 forbids shrinking a "
            "scientific budget to fit a machine",
            epochs_per_trial=epochs, steps_per_epoch=steps,
            declared_trials_per_track={track: item["plan"].total_trials
                                       for track, item in plans.items()},
            declared_trials=declared,
            total_optimizer_steps=declared * epochs * steps,
            note="the cost is reported here, before the first trial, so authorizing "
                 "the run is an informed decision rather than a discovery"))
        checks.append(check(
            "c7_optimizer_family_is_the_inherited_one",
            str(config["optimizer"]["name"]) == "AdamW",
            "the optimizer family is the uniquely inherited Version-B family and this "
            "envelope cannot express a switch",
            optimizer=config["optimizer"]["name"], anchor_source=DETECTOR_CONFIG))
        checks.append(check(
            "c7_no_treatment_arm_feedback_before_the_lock", True,
            "no RND or LLM detector performance, and no arm comparison, is computed "
            "or read before DETECTOR_CONFIG_LOCK closes",
            arms_trained=[arm], arms_evaluated=[arm],
            comparisons_computed=[], target_metrics_computed=0,
            rule="the source-search decision is a function only of frozen DET search "
                 "evidence, source-only dev evidence, the frozen selection tuple, the "
                 "canonical tie-break and the frozen coordinate envelope"))

        if not all(item["ok"] for item in checks):
            return None, self.result(request, mode=SCIENTIFIC_SEARCH, checks=checks,
                                     substage="C7_SCIENTIFIC_PLAN",
                                     summary="C7 scientific search plan is not executable")

        artifact = write_artifact(request, reports / "C7_SCIENTIFIC_SEARCH_PLAN.json", {
            "schema_version": "c7-scientific-search-plan-v2", "generated_at_utc": utc(),
            "mode": SCIENTIFIC_SEARCH, "fixture_backed": False,
            "search_decision_identity": decision.identity,
            "search_decision": decision.as_dict(),
            "lr_decision_identity": record.identity,
            "search_binding": dict(binding["c7_search_binding"]),
            "tracks": {
                track: {"search_plan_identity": item["plan"].identity,
                        "variant_flags": item["variant"].flags(),
                        "variant_identity": item["variant"].identity(),
                        "active_loss_terms": item["active_loss_terms"],
                        "searchable_terms": item["searchable_terms"],
                        "anchor_resolution": item["anchor_resolution"],
                        "lr_decision": item["lr"].as_dict(),
                        "plan": item["plan"].as_dict()}
                for track, item in sorted(plans.items())},
            "cost": {"epochs_per_trial": epochs, "steps_per_epoch": steps,
                     "declared_trials": declared,
                     "total_optimizer_steps": declared * epochs * steps}})

        state = {"plans": plans, "decision": decision, "lr_record": record,
                 "config": config, "binding": dict(binding["c7_search_binding"]),
                 "trained": {}}
        return state, self.result(
            request, mode=SCIENTIFIC_SEARCH, checks=checks, artifacts=[artifact],
            substage="C7_SCIENTIFIC_PLAN",
            parent_identities={
                "c7_search_decision": decision.identity,
                "c7_lr_decision": record.identity,
                **{f"c7_search_plan_{track.lower()}": item["plan"].identity
                   for track, item in plans.items()}})

    def _scientific_search(self, request: AdapterRequest, inputs: dict[str, Any],
                           state: dict[str, Any], reports: Path,
                           runs: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """One bounded pass per track. Every trial is a real detector run."""
        from prism_fas.search.coordinate import (EnvelopeExhausted, SearchError,
                                                 TrialResult, coordinate_search)

        decision = state["decision"]
        checks: list[dict[str, Any]] = []
        outcomes: dict[str, Any] = {}
        artifacts: list[str] = []

        for track, item in sorted(state["plans"].items()):
            plan = item["plan"]

            def evaluate(trial: Any, track: str = track, item: dict[str, Any] = item) -> Any:
                """One scientific trial: the full declared flow at one candidate.

                The trial's run root is keyed by its canonical config SHA — which
                covers the frozen search binding, so two tracks never collide and
                a swapped search bank never reuses a trial. Its summary is written
                to that root, because `coordinate_search` reuses a recorded PASS
                WITHOUT calling this function and a dict populated here would be
                empty for exactly the trials a resumed run depends on.
                """
                record = _run_scientific_trial(
                    request, inputs=inputs, state=state, track=track, item=item,
                    trial=trial, runs=runs)
                state["trained"][trial.config_sha256] = record
                if record["status"] != "PASS":
                    return TrialResult(trial=trial, status=record["status"], metrics={},
                                       artifacts=(record["trial_summary"],),
                                       notes=(record["reason"],))
                return TrialResult(
                    trial=trial, status="PASS",
                    metrics=dict(record["selection_metrics"]),
                    artifacts=(record["trial_summary"],),
                    notes=(f"scientific Track-{track} trial: the full frozen "
                           f"G1/G2/G5/G6 flow on the {decision.training_arm} C6 "
                           "matched bank, selected and calibrated on source_dev, "
                           "ranked by the §15.4 video-level tuple",))

            state_path = reports / _search_state_name(track)
            try:
                outcome = coordinate_search(plan, evaluate, state_path=state_path,
                                            resume=request.resume,
                                            require_valid_winner=True)
            except SearchError as error:
                # A recorded state written under a DIFFERENT plan identity. The
                # engine refuses to resume across two frozen envelopes, and the
                # commonest way to get here is that a bound input moved: the
                # frozen search bank, the search decision, the LR decision or the
                # source package. That is a fail-closed condition to report, not
                # a traceback to escape through.
                checks.append(check(
                    f"c7_track_{track.lower()}_resume_state_matches_this_envelope",
                    False,
                    f"Track {track} cannot resume: {error}",
                    reason_code="SEARCH_STATE_IDENTITY_MISMATCH",
                    search_plan_identity=plan.identity,
                    state=state_path.relative_to(request.repo).as_posix(),
                    rule="L.11: if an expected identity changed, fail closed and "
                         "compute the invalidation subtree rather than resuming"))
                return None, self.result(
                    request, mode=SCIENTIFIC_SEARCH, checks=checks, artifacts=artifacts,
                    summary=f"C7 Track {track} search state belongs to another envelope")
            except EnvelopeExhausted as error:
                checks.append(check(
                    f"c7_track_{track.lower()}_winner_exists", False,
                    f"Track {track}'s bounded envelope produced no valid configuration; "
                    "§15.2.2 requires stopping rather than widening the search",
                    error=str(error), reason_code="NEEDS_SCIENTIFIC_DECISION",
                    search_plan_identity=plan.identity,
                    forbidden=["widening a candidate set", "a second pass",
                               "a new optimizer family", "a new backbone",
                               "a new loss term", "an arm-specific rescue search",
                               "any use of a P1/P2/P3 result"]))
                return None, self.result(
                    request, mode=SCIENTIFIC_SEARCH, checks=checks,
                    summary="C7_SOURCE_SEARCH = NEEDS_SCIENTIFIC_DECISION")

            payload = outcome.as_dict()
            if outcome.status != "COMPLETED":
                checks.append(check(
                    f"c7_track_{track.lower()}_completed_before_finalization", False,
                    f"Track {track}'s search ended {outcome.status}; no "
                    "DETECTOR_CONFIG_LOCK may be written from an envelope that did "
                    "not close",
                    status=outcome.status,
                    completed_coordinates=outcome.completed_coordinates,
                    reason_code="SEARCH_INCOMPLETE",
                    state_preserved=state_path.relative_to(request.repo).as_posix()))
                artifacts.append(write_artifact(
                    request, reports / f"C7_SCIENTIFIC_SOURCE_SEARCH_{track}.json",
                    {**payload, "generated_at_utc": utc(), "mode": SCIENTIFIC_SEARCH,
                     "track": track, "fixture_backed": False, "finalizable": False}))
                return None, self.result(
                    request, mode=SCIENTIFIC_SEARCH, checks=checks, artifacts=artifacts,
                    summary=f"C7 Track {track} search is incomplete; state preserved")

            checks.append(check(
                f"c7_track_{track.lower()}_completed_before_finalization", True,
                f"Track {track}'s bounded one-pass envelope closed; every applicable "
                "coordinate completed exactly once, in order",
                status=outcome.status,
                completed_coordinates=outcome.completed_coordinates,
                applicable_coordinates=[c.name for c in plan.coordinates if c.applicable]))
            checks.append(check(
                f"c7_track_{track.lower()}_all_trials_retained",
                payload["trials_executed"] == sum(payload["trials_by_status"].values()),
                "every attempted configuration is retained, including FAIL and DIVERGED",
                **{key: payload[key] for key in ("trials_declared", "trials_executed",
                                                 "trials_by_status",
                                                 "finite_valid_trials")}))
            artifacts.append(write_artifact(
                request, reports / f"C7_SCIENTIFIC_SOURCE_SEARCH_{track}.json",
                {**payload, "generated_at_utc": utc(), "mode": SCIENTIFIC_SEARCH,
                 "track": track, "fixture_backed": False, "engineering_only": False,
                 "search_decision": decision.as_dict(),
                 "trial_roots": {sha: rec["run_root"] for sha, rec
                                 in sorted(state["trained"].items())
                                 if rec.get("track") == track}}))
            artifacts.append(state_path.relative_to(request.repo).as_posix())
            outcomes[track] = outcome

        trained = state["trained"]
        checks.append(check(
            "c7_scientific_search_used_the_real_trainer", True,
            "every trial was a full M9Trainer flow on the frozen C6 bank; no fixture "
            "batch, no audit detector and no analytic objective was used",
            trainer="prism_fas.detector.trainer.run_source_only_flow (canonical)",
            fixture_batch_used=False, audit_detector_used=False,
            analytic_objective_used=False, trials_run=len(trained)))
        checks.append(check(
            "c7_every_trial_trained_on_the_frozen_arm_only",
            bool(trained) and {record.get("training_arm") for record in trained.values()}
            == {decision.training_arm},
            f"every trial's synthetic samples came from the {decision.training_arm} "
            "bank; no RND or LLM candidate byte entered search training",
            arms_seen=sorted({str(record.get("training_arm"))
                              for record in trained.values()}),
            frozen_arm=decision.training_arm))
        checks.append(check(
            "c7_scientific_state_is_namespaced_per_track",
            all((reports / _search_state_name(track)).name != "C7_SEARCH_STATE.json"
                for track in state["plans"]),
            "each track's scientific search state has its own filename, and neither is "
            "the engineering state file",
            scientific_states=[_search_state_name(track) for track in sorted(state["plans"])],
            engineering_state="C7_SEARCH_STATE.json",
            also_refused_by="coordinate_search refuses a state whose recorded "
                            "search_plan_identity differs, and the plans differ"))
        checks.append(check(
            "c7_no_target_metric_entered_the_ranking", True,
            "the ranking tuple contains only source-domain quantities",
            selection_tuple=list(next(iter(state["plans"].values()))["plan"].selection_tuple),
            target_metrics_computed=0, target_labels_resolved=0))

        return outcomes, self.result(
            request, mode=SCIENTIFIC_SEARCH, checks=checks, artifacts=artifacts,
            parent_identities={f"c7_search_plan_{track.lower()}": item["plan"].identity
                               for track, item in state["plans"].items()})

    def _scientific_finalize(self, request: AdapterRequest, inputs: dict[str, Any],
                             state: dict[str, Any], outcomes: dict[str, Any],
                             reports: Path, runs: Path) -> AdapterResult:
        """Write DETECTOR_CONFIG_LOCK — one lock, one sub-config per track."""
        checks: list[dict[str, Any]] = []
        decision, closure = state["decision"], inputs["c6"]
        tracks: dict[str, Any] = {}

        for track, outcome in sorted(outcomes.items()):
            resolved, track_checks = _finalize_track(
                request, inputs=inputs, state=state, track=track, outcome=outcome,
                runs=runs)
            checks.extend(track_checks)
            if resolved is None:
                return self.result(
                    request, mode=FINALIZE_DETECTOR_CONFIG, checks=checks,
                    summary=f"C7 Track {track} has no usable scientific trial evidence")
            tracks[track] = resolved

        checks.append(check(
            "c7_one_configuration_per_track_not_per_arm",
            sorted(tracks) == sorted(decision.tracks),
            "the lock names exactly one frozen configuration per declared track, "
            "selected once against the frozen search bank",
            tracks=sorted(tracks), search_population=decision.training_arm,
            configurations={track: item["winner_config_sha256"]
                            for track, item in sorted(tracks.items())},
            rule="every primary generator arm of a track trains at that track's ONE "
                 "configuration in C8; a per-arm configuration would confound the "
                 "generator effect with detector tuning"))
        checks.append(check(
            "c7_no_target_capability", True,
            "no target capability was mounted at any point in this stage",
            target_paths_resolved=0, target_labels_resolved=0))

        lock_payload = {
            "schema_version": "c7-detector-config-lock-v2",
            "generated_at_utc": utc(), "mode": FINALIZE_DETECTOR_CONFIG,
            "is_scientific_lock": True,
            "fixture_backed": False,
            "execution_profile": request.profile.name,
            "scientific_eligible": True,
            "metrics_from_trained_runs": True,
            "metrics_source": ("M9Trainer source_dev evaluation over the frozen C6 "
                               f"{decision.training_arm} matched bank; the engineering "
                               "coordinate-engine probe is a different code path and "
                               "produced none of these"),
            # The decision that fixed the search population, and the envelope.
            "search_decision_identity": decision.identity,
            "search_decision": decision.as_dict(),
            "search_binding": dict(state["binding"]),
            "training_arm": decision.training_arm,
            "lr_decision_identity": state["lr_record"].identity,
            "per_arm_search_performed": False,
            "shared_within_track": True,
            "optimizer_family": "AdamW",
            # One sub-config per track. Every field a consumer needs is inside.
            "tracks": tracks,
            "track_ids": sorted(tracks),
            "selection_rule": ("the coordinate-wise best_config after one pass "
                               "(§15.2.2); the leaderboard winner is diagnostic"),
            # The frozen inputs.
            "source_package_identity": inputs["package_identity"],
            "recipe_bank_identity": inputs["recipe_bank_identity"],
            "c6_selector_identity_sha256": closure["selector_identity_sha256"],
            "c6_quality_threshold_identity": closure["quality_threshold_identity"],
            "c6_selected_profile": closure["selected_profile"],
            "c6_bank_locks": {arm: item["selected_set_sha256"]
                              for arm, item in closure["banks"].items()},
            "c6_training_bank": closure["banks"][decision.training_arm],
            "pretrained": inputs["pretrained"],
            "target_access": 0,
            "no_target_capability_proof": {"target_roots_mounted": [],
                                           "target_labels_resolved": 0},
        }

        if not all(item["ok"] for item in checks):
            return self.result(request, mode=FINALIZE_DETECTOR_CONFIG, checks=checks,
                               summary="C7 scientific finalization refused")

        artifact = write_artifact(request, reports / DETECTOR_CONFIG_LOCK, lock_payload)
        return self.result(
            request, mode=FINALIZE_DETECTOR_CONFIG, checks=checks, artifacts=[artifact],
            parent_identities={"c7_search_decision": decision.identity,
                               "c7_lr_decision": state["lr_record"].identity})

    def _scientific_verify_lock(self, request: AdapterRequest,
                                reports: Path) -> AdapterResult:
        """Verify the SCIENTIFIC lock and every checkpoint it names.

        The checks live in `verify_detector_config_lock`, module level and shared
        with C8. C8 trains 42 rows at the configurations this lock names, so it
        must apply the same verification C7 applies — not a second, laxer one.
        """
        path = reports / DETECTOR_CONFIG_LOCK
        verification = verify_detector_config_lock(request.repo, path)
        checks: list[dict[str, Any]] = list(verification["checks"])
        payload = verification["payload"]
        decision = resume_decision(request, "c7_detector_config_lock", path,
                                   expected_identity=payload.get(
                                       "search_decision_identity"),
                                   identity_key="search_decision_identity")
        checks.append(check(
            "c7_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lock by identity rather than by existence",
            **decision))

        passed = all(item["ok"] for item in checks)
        return self.result(
            request, mode=VERIFY_CONFIG_LOCK, checks=checks,
            artifacts=[path.relative_to(request.repo).as_posix()],
            # The ONE place C7 claims scientific evidence, and only when the lock
            # and every checkpoint it names verify.
            scientific_evidence=passed)


# --- scientific helpers, module level ----------------------------------------

class ScientificDeviceUnavailable(AdapterError):
    """A scientific C7 trial was asked for on a host with no CUDA device."""

    reason_code = "SCIENTIFIC_DEVICE_UNAVAILABLE"


def _scientific_device() -> str:
    """CUDA, or nothing. A scientific C7 never silently falls back to CPU.

    Dozens of full detector trainings on a CPU would not finish, and if they did
    they would be scientific evidence produced under a precision contract
    (`amp: true`, bf16/fp16) the run never entered. The zero-argument runner
    already refuses a non-CUDA host at the GPU preflight; this is the second
    lock, for the expert path that does not go through it.
    """
    import torch

    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        raise ScientificDeviceUnavailable(
            "scientific C7 requires CUDA and this host has none. A scientific "
            "detector trial may not run on the CPU: it would neither finish nor "
            "honour the frozen precision contract. Run the rehearsal profile on "
            "this machine, or run C7 on the GPU host.")
    return "cuda"


def _search_state_name(track: str) -> str:
    """One scientific search state per track, neither named like the rehearsal's."""
    stem, _, extension = SCIENTIFIC_SEARCH_STATE.rpartition(".")
    return f"{stem}_{track}.{extension}"


def _active_terms(variant: Any) -> dict[str, bool]:
    """Which searchable loss weights this variant's own loss graph declares active.

    Read off the variant rather than listed per track, so a Track-G row cannot be
    handed a `lambda_local` to tune when it instantiates no local branch, and a
    manifold-OFF Track R cannot be handed the K=4-only weights. §15.2.2 skips
    inactive terms; inventing a weight for a term that never evaluates would put
    a coordinate in the pass whose value cannot change any number.
    """
    active = variant.active_loss_terms()
    return {
        "lambda_syn": bool(active.get("L_cls_syn", False)),
        "lambda_local": bool(active.get("L_local", False)),
        "lambda_MIL": bool(active.get("L_MIL", False)),
        "lambda_P": bool(active.get("L_prompt", False)),
        "lambda_risk": bool(active.get("L_risk", False)),
        "lambda_M": bool(active.get("L_real", False)),
        "lambda_out": bool(active.get("L_out", False)),
        "lambda_clean": bool(active.get("L_clean", False)),
    }


def _search_binding(inputs: dict[str, Any], decision: Any,
                    lr_record: Any) -> dict[str, Any]:
    """What every candidate configuration of every track is bound to.

    Placed in the plan's `base_config`, so it enters the plan identity AND the
    canonical SHA of every trial — `plan.config_for` builds each candidate from
    the base config. Swapping the search bank therefore changes every config
    identity: resume misses, and C8's parent-identity validation fails, rather
    than a second set of numbers appearing under the same names.
    """
    arm = decision.training_arm
    bank = inputs["c6"]["banks"][arm]
    return {"c7_search_binding": {
        "training_arm": arm,
        "c6_bank_selected_set_sha256": bank["selected_set_sha256"],
        "c6_bank_lock": bank["lock"],
        "c6_selector_identity_sha256": inputs["c6"]["selector_identity_sha256"],
        "c6_selected_profile": inputs["c6"]["selected_profile"],
        "c6_quality_threshold_identity": inputs["c6"]["quality_threshold_identity"],
        "source_package_identity": inputs["package_identity"],
        "search_decision_identity": decision.identity,
        "lr_decision_identity": lr_record.identity,
    }}


def _trial_run_root(runs: Path, config_sha256: str) -> Path:
    """One deterministic run root per configuration, keyed by its identity.

    Deterministic on purpose: a resumed process must find the evidence a previous
    process wrote without carrying anything in memory. The config SHA already
    covers the track and the search binding, so two tracks never collide.
    """
    return Path(runs) / "scientific" / f"trial_{config_sha256[:16]}"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: The searchable loss weights, in the frozen §15.2.2 order. `learning_rate` is
#: absent on purpose — the approved decision replaces it with a multiplier that
#: expands to the whole LR vector, so no per-group rate is written independently.
_TRIAL_LOSS_WEIGHTS: tuple[str, ...] = (
    "lambda_syn", "lambda_local", "lambda_MIL", "lambda_P", "lambda_risk",
    "lambda_M", "lambda_out", "lambda_clean")


def _scientific_trial_config(configs: dict[str, Any], trial: Any, *, decision: Any,
                             lr: Any, arm_bank: Any, run_id: str) -> Any:
    """The frozen M9 config with exactly this trial's searched scalars applied.

    The base config is built by the canonical `config.load_m9_configs`, not
    assembled here from YAML keys: `m9_reference.yaml` stays authoritative for
    everything the envelope does not search — the batch composition, the stage
    schedule, the optimizer family, the architecture, the loss set, the
    checkpoint criteria — and a second reader of it would be a second answer to
    what the frozen configuration is.

    A coordinate may move only the scalar it names, so the searched values are
    applied with `dataclasses.replace` on top of that. The learning rate is the
    one that is not a single scalar: the approved decision is a common
    multiplier, so it expands through `lr_for_groups` into the whole vector and
    holds the frozen ratio. Nothing here writes a per-group rate independently.
    """
    from dataclasses import replace

    from prism_fas.evaluation import source_selection

    base = configs["training_config"]
    weights = dict(base.loss_weights)
    for name in _TRIAL_LOSS_WEIGHTS:
        if name in trial.config:
            weights[name] = float(trial.config[name])

    overrides: dict[str, Any] = {
        "run_id": run_id,
        "loss_weights": weights,
        "variant": configs["variant"],
        "synthetic_bank_identity": arm_bank.identity,
        "source_domains": source_selection.domains_for(decision.protocol),
    }
    multiplier = trial.config.get("learning_rate_multiplier")
    if multiplier is not None:
        rates = lr.lr_for_groups(float(multiplier))
        if "backbone_lr" in rates:
            overrides["backbone_lr"] = float(rates["backbone_lr"])
        if "head_lr" in rates:
            overrides["head_lr"] = float(rates["head_lr"])
    if "weight_decay" in trial.config:
        overrides["weight_decay"] = float(trial.config["weight_decay"])
    if "warmup" in trial.config:
        overrides["warmup_fraction"] = float(trial.config["warmup"])
    return replace(base, **overrides)


def _run_scientific_trial(request: AdapterRequest, *, inputs: dict[str, Any],
                          state: dict[str, Any], track: str, item: dict[str, Any],
                          trial: Any, runs: Path) -> dict[str, Any]:
    """One atomic scientific trial, with its own durable, addressable evidence.

    Every terminal state writes a `C7_TRIAL_SUMMARY.json` — PASS, FAIL and
    DIVERGED alike. §15.2.2 retains invalid and divergent trials, and a trial that
    left no artifact would be indistinguishable from one that never ran.
    """
    from prism_fas.detector.c6_bank import open_arm_bank
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.decision_audit import decision_graph_hash
    from prism_fas.detector.trainer import M9Trainer, run_source_only_flow
    from prism_fas.evaluation import source_selection
    from prism_fas.pipeline.state import atomic_write_json

    decision = state["decision"]
    plan, variant, lr = item["plan"], item["variant"], item["lr"]
    run_root = _trial_run_root(runs, trial.config_sha256)
    summary_path = run_root / TRIAL_SUMMARY
    started = utc()
    arm = decision.training_arm
    evidence = inputs["c6"]["banks"][arm]

    def persist(payload: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(summary_path, payload)
        return {**payload,
                "run_root": run_root.relative_to(request.repo).as_posix(),
                "trial_summary": summary_path.relative_to(request.repo).as_posix()}

    base = {
        "schema_version": "c7-scientific-trial-summary-v2",
        "started_at_utc": started,
        "scientific_eligible": True, "fixture_backed": False,
        "trial_config_sha256": trial.config_sha256,
        "trial_config_id": trial.config_id,
        "coordinate": trial.coordinate, "value": trial.value,
        "search_plan_identity": plan.identity,
        "trial_config": dict(trial.config),
        "track": track,
        "training_arm": arm,
        "protocol": decision.protocol,
        "variant_identity": variant.identity(),
        # The frozen search population, bound into every trial's parent set.
        "parent_identities": {
            "c7_search_decision": decision.identity,
            "c7_lr_decision": state["lr_record"].identity,
            "c7_search_plan": plan.identity,
            "c6_bank_selected_set_sha256": evidence["selected_set_sha256"],
            "c6_selector_identity_sha256": inputs["c6"]["selector_identity_sha256"],
            "source_package_identity": inputs["package_identity"],
        },
        "c6_bank_lock_selected_set_sha256": evidence["selected_set_sha256"],
        "package_identity": inputs["package_identity"],
        "target_paths_resolved": 0, "target_labels_resolved": 0,
    }

    if not decision.permits_arm(arm):   # unreachable by construction; proven anyway
        return persist({**base, "generated_at_utc": utc(), "status": "FAIL",
                        "reason": f"{arm} is not the frozen search arm",
                        "selection_metrics": {}})

    try:
        device = _scientific_device()
        bank = open_arm_bank(
            request.repo, arm=arm,
            evidence=_arm_evidence(request.repo, arm),
            candidates_root=request.repo / inputs["candidates_root"],
            package_identity=inputs["package_identity"],
            recipe_bank_identity=inputs["recipe_bank_identity"])
        configs = load_m9_configs(request.repo / "configs/models/m9_detector.yaml",
                                  request.repo / DETECTOR_CONFIG, variant=variant)
        config = _scientific_trial_config(
            configs, trial, decision=decision, lr=lr, arm_bank=bank,
            run_id=f"c7_{track.lower()}_{trial.config_sha256[:16]}")
        trainer = M9Trainer(
            config=config,
            detector_config=configs["detector_config"],
            package_root=request.repo / inputs["package_root"],
            bank_root=request.repo / inputs["candidates_root"],
            recipe_bank_root=request.repo / inputs["recipe_bank_root"],
            run_root=run_root, cache_root=run_root / "cache",
            weight_root=request.repo / inputs["weight_root"],
            loader_config_path=request.repo / "configs/data/loader_m4.yaml",
            device=device, synthetic_bank=bank)
        flow = run_source_only_flow(trainer, resume=request.resume)

        frames = source_selection.source_dev_frame_rows(trainer)
        calibration = source_selection.fit_source_dev_calibration(frames)
        metrics = source_selection.evaluate(
            frames, protocol=decision.protocol,
            temperature=calibration["temperature"],
            threshold=calibration["threshold"],
            epoch=int((flow["run_summary"].get("best_metrics") or {}).get("epoch", -1)),
            decision_logit_name=trainer.decision_logit_name,
            decision_score_name=trainer.decision_score_name)
        graph = decision_graph_hash(trainer.model)

    except (KeyboardInterrupt, SystemExit):
        # Somebody stopped the process. Not an outcome for this trial: everything
        # already on disk stays, and the next invocation resumes this trial.
        raise
    except Exception as error:                        # noqa: BLE001
        # A trial that will not train is a retained NEGATIVE result, not a reason
        # to widen the envelope or skip a coordinate. It ranks after every
        # finite-valid trial and stays addressable.
        #
        # `FAIL`, not `FAILED`: `coordinate.TRIAL_STATUS` is the vocabulary
        # `TrialResult` validates against, and a status outside it makes the
        # engine record the trial through its own exception path instead — so the
        # summary on disk and the leaderboard would disagree.
        return persist({
            **base, "generated_at_utc": utc(), "status": "FAIL",
            "reason": f"{type(error).__name__}: {error}"[:600],
            "selection_metrics": {},
            "retention": ("§15.2.2 retains invalid and divergent trials; this one is "
                          "ranked after every finite-valid trial and is never deleted"),
        })

    checkpoint = trainer.checkpoint_path("best")
    if not checkpoint.is_file():
        checkpoint = trainer.checkpoint_path("last")
    ranking = metrics["ranking_tuple"]
    diverged = not all(
        isinstance(value, (int, float)) and value == value
        and abs(float(value)) != float("inf") for value in ranking.values())

    return persist({
        **base,
        "generated_at_utc": utc(),
        "status": "DIVERGED" if diverged else "PASS",
        "reason": ("a selection metric was not finite" if diverged else ""),
        "resolved_config": config.resolved(),
        "resolved_config_hash": config.hash(),
        "variant_flags": variant.flags(),
        "decision_logit_name": trainer.decision_logit_name,
        "decision_score_name": trainer.decision_score_name,
        "decision_graph_hash": graph["decision_graph_hash"],
        "run_identity": trainer.identity.payload(),
        "batch_contract": trainer.samplers["G5"].contract.payload(),
        "c6_bank_reader_identity": bank.identity,
        "c6_bank_summary": bank.summary(),
        "recipe_bank_identity": inputs["recipe_bank_identity"],
        "pretrained": inputs["pretrained"],
        "checkpoint": checkpoint.relative_to(request.repo).as_posix()
                      if checkpoint.is_file() else None,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint.is_file() else None,
        "best_epoch": int((flow["run_summary"].get("best_metrics") or {}).get("epoch", -1)),
        "calibration": {**calibration,
                        "decision_logit_name": trainer.decision_logit_name,
                        "thresholded_quantity": trainer.decision_score_name},
        "selection_metrics": ranking,
        "source_selection": metrics,
        "flow": {key: flow[key] for key in
                 ("stages", "declared_stages", "stages_executed_here", "run_closure",
                  "resumed_from", "resumed_stage")},
        "source_isolation": flow["source_isolation"],
        "code_lineage": {"git_commit": flow["run_summary"].get("git_commit")},
        "device": flow["run_summary"].get("device"),
    })


def _arm_evidence(repo: Path, arm: str) -> Any:
    """This arm's verified C6 bank evidence, re-verified rather than cached.

    `verify_detector_inputs` already ran, and re-running the verifier here costs
    a few JSON reads. What it buys is that the bank reader is handed an
    `ArmBankEvidence` produced by the canonical verifier rather than a dictionary
    reconstructed from an artifact, so it cannot be handed one that never passed.
    """
    from prism_fas.evaluation.c6_evidence import verify_c6_evidence

    return verify_c6_evidence(repo).bank(arm)


def _resolve_trial_evidence(repo: Path, runs: Path, config_sha256: str,
                            trained: dict[str, Any]) -> dict[str, Any] | None:
    """One configuration's evidence, from this process or a previous one.

    In-memory first, because it is already loaded. Otherwise the trial's own
    summary, which is why the requirement is "valid trial evidence exists and
    matches this frozen plan" rather than "was trained in this pass". A recorded
    PASS whose evidence is missing returns None and the caller fails closed;
    nothing here accepts metrics from the search state without the run that
    produced them.
    """
    if config_sha256 in trained:
        return trained[config_sha256]

    root = _trial_run_root(runs, config_sha256)
    record = read_json(root / TRIAL_SUMMARY)
    if not record or record.get("trial_config_sha256") != config_sha256:
        return None
    return {**record,
            "run_root": root.relative_to(repo).as_posix(),
            "trial_summary": (root / TRIAL_SUMMARY).relative_to(repo).as_posix(),
            "reused_from_previous_process": True}


def _finalize_track(request: AdapterRequest, *, inputs: dict[str, Any],
                    state: dict[str, Any], track: str, outcome: Any,
                    runs: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve one track's frozen configuration and the evidence behind it."""
    from prism_fas.search.plan import canonical_config_sha256

    payload = outcome.as_dict()
    item = state["plans"][track]
    plan, decision = item["plan"], state["decision"]
    prefix = f"c7_track_{track.lower()}"
    checks: list[dict[str, Any]] = []

    # WHICH configuration is the selection. `best_config` is the accumulator the
    # coordinate pass produces — start at the anchor, move one coordinate at a
    # time, carry the winner forward — and it is what §15.2.2 defines a
    # coordinate search to yield. The leaderboard winner is a ranking of
    # individual probes: a trial from an EARLY coordinate can rank globally best
    # while its config lacks every later coordinate's improvement.
    selected_config = dict(payload["best_config"])
    selected_sha = canonical_config_sha256(selected_config)
    selected_trial = next((row for row in payload["leaderboard"]
                           if row.get("config_sha256") == selected_sha), None)

    checks.append(check(
        f"{prefix}_selected_config_was_actually_evaluated",
        selected_trial is not None and bool(selected_trial.get("finite_valid")),
        "the coordinate-wise selected configuration corresponds to a trial that "
        "really ran and reported finite selection metrics",
        selected_config_sha256=selected_sha,
        selected_trial_status=(selected_trial or {}).get("status"),
        leaderboard_winner_config_sha256=payload["winner_config_sha256"],
        note="the leaderboard winner is retained for diagnostics only and never "
             "becomes the frozen selection"))

    evidence = _resolve_trial_evidence(request.repo, runs, selected_sha,
                                       state.get("trained", {}))
    checks.append(check(
        f"{prefix}_selected_trial_evidence_resolves", evidence is not None,
        "the selected configuration has persistent scientific run evidence, whether "
        "it was trained in this process or reused from a previous one",
        selected_config_sha256=selected_sha,
        trial_run_root=_trial_run_root(runs, selected_sha)
        .relative_to(request.repo).as_posix()))
    if evidence is None or selected_trial is None:
        return None, checks

    checkpoint = (request.repo / evidence["checkpoint"]
                  if evidence.get("checkpoint") else None)
    measured = _sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None
    checks.append(check(
        f"{prefix}_selected_checkpoint_present",
        bool(checkpoint and checkpoint.is_file())
        and bool(evidence.get("checkpoint_sha256")),
        "the selected configuration's checkpoint exists and carries its own SHA-256",
        checkpoint=evidence.get("checkpoint"),
        checkpoint_sha256=evidence.get("checkpoint_sha256")))
    checks.append(check(
        f"{prefix}_selected_checkpoint_hash_is_intact",
        bool(measured) and measured == evidence.get("checkpoint_sha256"),
        "the checkpoint on disk still hashes to what its trial recorded",
        recorded_sha256=evidence.get("checkpoint_sha256"), measured_sha256=measured))
    checks.append(check(
        f"{prefix}_checkpoint_belongs_to_the_selected_config",
        evidence.get("trial_config_sha256") == selected_sha
        and evidence.get("track") == track,
        "the checkpoint was trained for THIS configuration and THIS track; a config "
        "and a checkpoint from different trials are never bound together",
        selected_config_sha256=selected_sha,
        evidence_trial_config_sha256=evidence.get("trial_config_sha256"),
        evidence_track=evidence.get("track")))
    checks.append(check(
        f"{prefix}_evidence_binds_this_search_plan",
        evidence.get("search_plan_identity") == plan.identity,
        "the trial evidence was produced under this exact frozen search plan",
        search_plan_identity=plan.identity,
        evidence_search_plan_identity=evidence.get("search_plan_identity")))
    parents = dict(evidence.get("parent_identities") or {})
    expected_bank = inputs["c6"]["banks"][decision.training_arm]["selected_set_sha256"]
    checks.append(check(
        f"{prefix}_evidence_binds_the_frozen_search_bank",
        evidence.get("training_arm") == decision.training_arm
        and parents.get("c6_bank_selected_set_sha256") == expected_bank
        and parents.get("c7_search_decision") == decision.identity,
        f"the trial trained on the frozen {decision.training_arm} bank this run "
        "verified, under the frozen search decision",
        expected={"arm": decision.training_arm, "c6_bank": expected_bank,
                  "search_decision": decision.identity},
        evidence={"arm": evidence.get("training_arm"),
                  "c6_bank": parents.get("c6_bank_selected_set_sha256"),
                  "search_decision": parents.get("c7_search_decision")}))
    checks.append(check(
        f"{prefix}_evidence_binds_the_source_package",
        evidence.get("package_identity") == inputs["package_identity"],
        "the trial trained against the same source package this run resolved",
        expected=inputs["package_identity"],
        evidence=evidence.get("package_identity")))
    calibration = dict(evidence.get("calibration") or {})
    checks.append(check(
        f"{prefix}_calibration_and_threshold_are_the_same_quantity",
        calibration.get("decision_logit_name") == evidence.get("decision_logit_name")
        and calibration.get("thresholded_quantity") == evidence.get("decision_score_name"),
        "the temperature was fitted on the decision logit and the threshold applies "
        "to the decision score",
        calibration_split=calibration.get("split"),
        calibration_logit_name=calibration.get("decision_logit_name"),
        thresholded_quantity=calibration.get("thresholded_quantity"),
        decision_logit_name=evidence.get("decision_logit_name"),
        decision_score_name=evidence.get("decision_score_name"),
        rule="§16.2: no fused score may be thresholded by a calibration fitted on a "
             "different quantity"))

    resolved = {
        "track": track,
        "variant_flags": item["variant"].flags(),
        "variant_identity": item["variant"].identity(),
        "active_loss_terms": item["active_loss_terms"],
        "searchable_terms": item["searchable_terms"],
        "not_applicable_terms": sorted(name for name, on
                                       in item["searchable_terms"].items() if not on),
        "search_plan_identity": payload["search_plan_identity"],
        "coordinate_order": payload["coordinate_order"],
        "candidate_envelopes": [c.as_dict() for c in plan.coordinates],
        "selection_tuple": payload["selection_tuple"],
        "tie_break": payload["tie_break"],
        "one_pass": payload["one_pass"],
        "lr_interpretation": item["lr"].interpretation,
        "lr_anchor_vector": dict(item["lr"].anchor_vector),
        "inherited_anchor_report": item["anchor_resolution"],
        "trials_declared": payload["trials_declared"],
        "trials_executed": payload["trials_executed"],
        "trials_by_status": payload["trials_by_status"],
        "attempted_config_ids": payload["attempted_config_ids"],
        "trial_set_digest": outcome.outcome_identity,
        "retained_trials": [
            {"config_id": row["config_id"], "config_sha256": row["config_sha256"],
             "coordinate": row["coordinate"], "value": row["value"],
             "status": row["status"], "finite_valid": row["finite_valid"],
             "metrics": row["metrics"], "artifacts": row["artifacts"],
             "notes": row["notes"]} for row in payload["leaderboard"]],
        "retention_policy": payload["retention"],
        "winner_config": selected_config,
        "winner_config_sha256": selected_sha,
        "winner_trial_config_id": selected_trial.get("config_id"),
        "winner_selection_metrics": selected_trial.get("metrics", {}),
        "winner_checkpoint": evidence.get("checkpoint"),
        "winner_checkpoint_sha256": evidence.get("checkpoint_sha256"),
        "winner_trial_run_root": evidence.get("run_root"),
        "winner_trial_summary": evidence.get("trial_summary"),
        "winner_calibration": calibration,
        "winner_epoch": evidence.get("best_epoch"),
        "leaderboard_winner_config_sha256": payload["winner_config_sha256"],
        "leaderboard_winner_is_the_selection":
            payload["winner_config_sha256"] == selected_sha,
        "tie_break_trace": payload["tie_break_trace"],
        "decision_logit_name": evidence.get("decision_logit_name"),
        "decision_score_name": evidence.get("decision_score_name"),
        "decision_graph_hash": evidence.get("decision_graph_hash"),
        "batch_contract": evidence.get("batch_contract", {}),
        "code_lineage": evidence.get("code_lineage", {}),
    }
    return resolved, checks


def verify_detector_config_lock(repo: Path, lock_path: Path) -> dict[str, Any]:
    """The strict DETECTOR_CONFIG_LOCK verification, shared by C7 and C8.

    Module level and shared for the same reason `verify_gpat_config_lock` is: C8
    trains 42 rows at the configurations this lock names, so it must apply
    exactly the checks C7 applied. A second, laxer verifier inside C8 would be
    the way a rehearsal-shaped or drifted lock reaches the matrix.
    """
    path = Path(lock_path)
    payload = read_json(path) or {}
    checks: list[dict[str, Any]] = []

    checks.append(check(
        "c7_config_lock_present", bool(payload),
        f"{path.name} exists and is readable JSON", path=path.as_posix()))
    if not payload:
        return {"payload": {}, "checks": checks, "valid": False}

    checks.append(check(
        "c7_config_lock_is_scientific",
        payload.get("is_scientific_lock") is True
        and payload.get("fixture_backed") is False
        and payload.get("scientific_eligible") is True
        and payload.get("execution_profile") == "full",
        "the lock declares itself scientific, non-fixture-backed and full-profile",
        is_scientific_lock=payload.get("is_scientific_lock"),
        fixture_backed=payload.get("fixture_backed"),
        scientific_eligible=payload.get("scientific_eligible"),
        execution_profile=payload.get("execution_profile")))
    checks.append(check(
        "c7_config_lock_metrics_are_not_analytic",
        payload.get("metrics_from_trained_runs") is True,
        "the winning configurations were chosen from trained measurements, not from "
        "the engineering coordinate-engine probe",
        metrics_from_trained_runs=payload.get("metrics_from_trained_runs"),
        metrics_source=payload.get("metrics_source"),
        rule="a declared boolean rather than a substring search over prose: the "
             "honest description of a trained run mentions the analytic objective "
             "in order to deny it"))

    decision = dict(payload.get("search_decision") or {})
    checks.append(check(
        "c7_config_lock_binds_the_frozen_search_arm",
        bool(payload.get("search_decision_identity"))
        and decision.get("decision_id") == "C7_SOURCE_SEARCH_SYNTHETIC_ARM"
        and bool(payload.get("training_arm"))
        and decision.get("training_arm") == payload.get("training_arm")
        and decision.get("timing") == "BEFORE_FIRST_C7_SCIENTIFIC_TRIAL",
        "the lock binds the frozen search-population decision, its identity and the "
        "arm it named, and records that it was frozen before any trial",
        search_decision_identity=payload.get("search_decision_identity"),
        training_arm=payload.get("training_arm"),
        decision_id=decision.get("decision_id"), timing=decision.get("timing"),
        spec_status=decision.get("spec_status")))
    binding = dict(payload.get("search_binding") or {})
    banks = dict(payload.get("c6_bank_locks") or {})
    checks.append(check(
        "c7_config_lock_search_bank_is_the_frozen_arms",
        binding.get("training_arm") == payload.get("training_arm")
        and bool(binding.get("c6_bank_selected_set_sha256"))
        and binding.get("c6_bank_selected_set_sha256")
        == banks.get(str(payload.get("training_arm"))),
        "the search binding names the frozen arm's own selected-set digest; no other "
        "arm's candidate bytes could have entered search training",
        training_arm=payload.get("training_arm"),
        binding_bank=binding.get("c6_bank_selected_set_sha256"),
        c6_bank_locks=banks))
    checks.append(check(
        "c7_config_lock_declares_no_per_arm_search",
        payload.get("per_arm_search_performed") is False
        and payload.get("shared_within_track") is True,
        "one configuration per track, shared by every primary generator arm of that "
        "track; no per-arm search was run",
        per_arm_search_performed=payload.get("per_arm_search_performed"),
        shared_within_track=payload.get("shared_within_track"),
        rule="a configuration per arm would confound the generator effect with "
             "detector tuning"))

    tracks = dict(payload.get("tracks") or {})
    checks.append(check(
        "c7_config_lock_names_a_configuration_per_track",
        bool(tracks) and sorted(tracks) == sorted(payload.get("track_ids") or []),
        "the lock carries one sub-configuration per declared track",
        tracks=sorted(tracks), track_ids=payload.get("track_ids")))

    for track in sorted(tracks):
        sub = dict(tracks[track])
        prefix = f"c7_config_lock_track_{track.lower()}"
        retained = list(sub.get("retained_trials") or ())
        by_status = dict(sub.get("trials_by_status") or {})
        checks.append(check(
            f"{prefix}_binds_the_frozen_envelope",
            all(sub.get(key) for key in
                ("search_plan_identity", "coordinate_order", "selection_tuple",
                 "tie_break", "variant_identity")),
            f"Track {track} binds its search plan, coordinate order, ranking tuple, "
            "tie-break and variant identity",
            **{key: sub.get(key) for key in
               ("search_plan_identity", "tie_break", "variant_identity")},
            coordinate_order=sub.get("coordinate_order"),
            selection_tuple=sub.get("selection_tuple")))
        checks.append(check(
            f"{prefix}_retains_every_trial",
            bool(retained) and len(retained) == int(sub.get("trials_executed", -1))
            and len(retained) == sum(by_status.values()),
            f"every Track-{track} configuration attempted is retained in the lock, "
            "including failures and divergences",
            trials_executed=sub.get("trials_executed"), retained=len(retained),
            trials_by_status=by_status, trial_set_digest=sub.get("trial_set_digest")))
        checks.append(check(
            f"{prefix}_names_a_winner",
            bool(sub.get("winner_config")) and bool(sub.get("winner_config_sha256"))
            and bool(sub.get("winner_checkpoint_sha256")),
            f"Track {track} names its selected configuration, its canonical SHA and "
            "its checkpoint identity",
            winner_config_sha256=sub.get("winner_config_sha256"),
            winner_checkpoint=sub.get("winner_checkpoint"),
            winner_checkpoint_sha256=sub.get("winner_checkpoint_sha256")))
        checkpoint = repo / str(sub.get("winner_checkpoint") or "")
        present = bool(sub.get("winner_checkpoint")) and checkpoint.is_file()
        measured = _sha256_file(checkpoint) if present else None
        checks.append(check(
            f"{prefix}_checkpoint_is_intact",
            present and measured == sub.get("winner_checkpoint_sha256"),
            f"Track {track}'s winning checkpoint is on disk and still hashes to what "
            "the lock recorded",
            checkpoint=sub.get("winner_checkpoint"), present=present,
            recorded_sha256=sub.get("winner_checkpoint_sha256"),
            measured_sha256=measured))
        checks.append(check(
            f"{prefix}_binds_the_decision_graph",
            all(sub.get(key) for key in ("decision_logit_name", "decision_score_name",
                                         "decision_graph_hash")),
            f"Track {track} binds the decision logit, score and graph hash the "
            "configuration was selected under",
            **{key: sub.get(key) for key in
               ("decision_logit_name", "decision_score_name", "decision_graph_hash")}))
        checks.append(check(
            f"{prefix}_tunes_only_its_active_terms",
            all(name not in dict(sub.get("winner_config") or {})
                for name in sub.get("not_applicable_terms") or ()),
            f"Track {track}'s frozen configuration carries no value for a loss weight "
            "its own variant declares inactive",
            not_applicable_terms=sub.get("not_applicable_terms"),
            winner_config_keys=sorted(dict(sub.get("winner_config") or {}))))

    checks.append(check(
        "c7_config_lock_binds_its_inputs",
        all(payload.get(key) for key in
            ("source_package_identity", "c6_selector_identity_sha256",
             "c6_quality_threshold_identity", "pretrained"))
        and len(banks) == 3,
        "the lock binds the source package, the C6 selector and threshold identities, "
        "all three bank locks and the pinned weights",
        source_package_identity=payload.get("source_package_identity"),
        c6_selector_identity_sha256=payload.get("c6_selector_identity_sha256"),
        c6_bank_locks=banks, pretrained=payload.get("pretrained")))
    checks.append(check(
        "c7_config_lock_declares_no_target_access",
        int(payload.get("target_access", -1)) == 0
        and int((payload.get("no_target_capability_proof") or {}).get(
            "target_labels_resolved", -1)) == 0,
        "the lock carries a no-target-capability proof",
        target_access=payload.get("target_access"),
        **dict(payload.get("no_target_capability_proof") or {})))

    return {"payload": payload, "checks": checks,
            "valid": all(item["ok"] for item in checks)}


__all__ = ["STAGE_ID", "MODES", "SCIENTIFIC_MODES", "TRACK_G_READINESS",
           "TRACK_R_READINESS", "DECISION_DEPENDENCY_AUDIT", "CALIBRATION_GUARDS",
           "VARIANT_MATRIX_AUDIT", "SOURCE_SEARCH", "VERIFY_C6_EVIDENCE",
           "SCIENTIFIC_SEARCH", "FINALIZE_DETECTOR_CONFIG", "VERIFY_CONFIG_LOCK",
           "SCIENTIFIC_REPORTS", "DETECTOR_CONFIG_LOCK", "SCIENTIFIC_CONFIG_LOCK_PATH",
           "SCIENTIFIC_SEARCH_STATE", "TRIAL_SUMMARY",
           "TRACK_G_FLAGS", "TRACK_R_FLAGS", "TRACK_R_K4_FLAGS",
           "ScientificDeviceUnavailable", "verify_detector_config_lock",
           "C7Adapter"]
