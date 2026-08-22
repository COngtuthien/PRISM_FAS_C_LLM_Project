"""C4 — neutral GPAT support, readiness and the bounded source search.

C4's scientific job is to train one generator-neutral GPAT once and freeze it.
This adapter makes that job *executable*: it drives the real model, the real
losses, the real checkpoint format and the real search engine on tiny fixtures,
so the defects that would otherwise surface after hours of GPU time surface here
in seconds.

Six modes, in the order a full pass would use them:

``PREPARE_SUPPORT``   build the conditioning support batch
``VALIDATE_SUPPORT``  check it against the frozen GPAT batch contract
``SMOKE_GPAT``        instantiate, forward, finite loss, backward, step, checkpoint, resume
``SOURCE_SEARCH``     execute the §15.2.3 one-pass coordinate envelope
``FINALIZE_GPAT``     record the search outcome and the configuration it selected
``VERIFY_LOCK``       verify the resulting lock re-derives from its own material

Three substitutions the smoke path makes, each recorded on the artifact rather
than described here and forgotten:

* the identity backbone (AdaFace) is absent from this machine, so the identity
  embedding is a deterministic stand-in. The identity *loss path* still runs;
  its *value* is not a measurement.
* the support pairs come from the frozen C3 recipe banks rather than from
  preprocessed source imagery, because the source packages are not built here.
  That is a deliberate integration: it proves the real 41-D conditioning vector
  computed from a real scientific recipe flows into the real generator.
* images are synthetic noise fields, not faces.

None of the three changes the model topology, the loss graph or the tensor
contract, which is exactly the boundary L.12 draws around what smoke may reduce.
Under `full` none of them is permitted: the precondition gate blocks first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import (AdapterError, AdapterRequest,
                                         AdapterResult)
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, import_canonical,
                                                read_json, resume_decision,
                                                stage_reports_dir, stage_runs_dir, utc,
                                                write_artifact)
from prism_fas.pipeline.execution import ExecutionContext

STAGE_ID = "C4"

PREPARE_SUPPORT = "PREPARE_SUPPORT"
VALIDATE_SUPPORT = "VALIDATE_SUPPORT"
SMOKE_GPAT = "SMOKE_GPAT"
SOURCE_SEARCH = "SOURCE_SEARCH"
FINALIZE_GPAT = "FINALIZE_GPAT"
VERIFY_LOCK = "VERIFY_LOCK"

MODES: tuple[str, ...] = (PREPARE_SUPPORT, VALIDATE_SUPPORT, SMOKE_GPAT,
                          SOURCE_SEARCH, FINALIZE_GPAT, VERIFY_LOCK)

GPAT_CONFIG = "configs/synthesis/gpat_m8.yaml"
#: The support conditioning is drawn from a frozen scientific bank so the
#: rehearsal exercises the real conditioning vector rather than a made-up one.
SUPPORT_RECIPE_SOURCE = "assets/recipe_banks/c3/llm/recipes.jsonl"

#: Kept small deliberately: 224x224 forward+backward on CPU is the cost driver,
#: and the control path is identical at two samples and at two thousand.
SMOKE_BATCH = 2


def _load_config(repo: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((repo / GPAT_CONFIG).read_text(encoding="utf-8"))


def _support_conditioning(repo: Path, count: int) -> tuple[Any, list[str]]:
    """The 41-D conditioning vectors for `count` frozen scientific recipes.

    Delegates to `prism_fas.recipes.conditioning`, which owns the feature order
    and the normalization. Building the vector here would be a second encoder.
    """
    import numpy as np

    from prism_fas.recipes.canonical import recipe_hash
    from prism_fas.recipes.conditioning import conditioning_vector
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    ontology = load_ontology(repo / "configs/recipes/ontology_m7.yaml")
    lines = (repo / SUPPORT_RECIPE_SOURCE).read_text(encoding="utf-8").splitlines()
    vectors, ids = [], []
    for line in lines[:count]:
        recipe = parse_recipe(json.loads(line))
        vectors.append(conditioning_vector(recipe, ontology))
        ids.append(recipe_hash(recipe))
    return np.stack(vectors), ids


def _fixture_batch(repo: Path, size: int, *, seed: int = 20260806) -> Any:
    """One valid GPAT batch built from frozen recipes and deterministic noise.

    Every tensor satisfies the real `GPATBatch` contract — sizes, binary masks,
    non-zero support, 41-D conditioning, 512-D identity embedding — because the
    contract is what we are trying to exercise. A batch that had to be waved
    past `validate()` would prove nothing.
    """
    import torch

    from prism_fas.synthesis.gpat_contracts import GPATBatch

    generator = torch.Generator().manual_seed(seed)
    conditioning, recipe_ids = _support_conditioning(repo, size)

    def image() -> torch.Tensor:
        return torch.rand(size, 3, 224, 224, generator=generator)

    # A centred rectangular support with real interior and real exterior: the
    # outside-mask invariant is only meaningful when something is outside.
    support = torch.zeros(size, 1, 224, 224)
    support[:, :, 56:168, 56:168] = 1.0
    style = torch.zeros(size, 1, 224, 224)
    style[:, :, 40:184, 40:184] = 1.0

    return GPATBatch(
        live_image=image(),
        source_spoof_image=image(),
        recipe_conditioning=torch.tensor(conditioning, dtype=torch.float32),
        target_support_mask=support,
        source_style_mask=style,
        recipe_strength=torch.full((size,), 0.5),
        # Stands in for the frozen AdaFace embedding, which is not resolvable
        # here. Deterministic and L2-normalized so the identity loss is finite
        # and reproducible; it is a fixture, not a measurement.
        live_identity_embedding=torch.nn.functional.normalize(
            torch.randn(size, 512, generator=generator), dim=-1),
        pair_ids=tuple(f"smoke-pair-{index:03d}" for index in range(size)),
        recipe_ids=tuple(recipe_ids)).validate()


def _identity_stand_in(batch: Any, output: Any) -> Any:
    """A deterministic generated-image embedding for the identity loss path.

    The real embedding comes from the frozen AdaFace tower. Absent that, this
    keeps the loss *graph* connected to the generated image — so the backward
    pass really does flow through the identity term — while making no claim
    about identity preservation. It is a differentiable projection, not a face
    recognizer, and the artifact says so.
    """
    import torch

    pooled = output.synthetic_image.mean(dim=(2, 3))          # [B,3], differentiable
    reference = batch.live_identity_embedding                  # [B,512]
    return torch.nn.functional.normalize(
        reference + 0.01 * pooled.repeat(1, 512 // 3 + 1)[:, :512], dim=-1)


@dataclass
class C4Adapter(EngineeringAdapter):
    """The C4 execution adapter. Thin: every model and loss is imported."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Neutral GPAT support and final checkpoint"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = True

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("gpat_config", GPAT_CONFIG,
                          "the frozen GPAT training configuration"),
            RequiredInput("c3_recipe_banks", "assets/recipe_banks/c3",
                          "the frozen C3 scientific recipe banks the support conditions on"),
            RequiredInput("source_packages", "data/packages",
                          "preprocessed source packages supplying the live/spoof pairs"),
            RequiredInput("gpat_pair_plan", "data/packages/gpat_pairs",
                          "the frozen source-only GPAT pair plan"),
        )

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        """Two workflows, chosen by the context — never one that adapts.

        The defect this branch closes: there was a single workflow, built as an
        engineering rehearsal, and `--profile full` ran it. It evaluated each
        search trial with ONE optimizer step on a `_fixture_batch`, scored it
        with `_identity_stand_in` instead of the frozen AdaFace, called
        `coordinate_search(require_valid_winner=False)`, and finished by writing
        `C4_ENGINEERING_CONFIG_RECORD.json` and asserting that the scientific
        `GPAT_CONFIG_LOCK.json` did NOT exist. Every check passed, because the
        engineering path is correct engineering — which is exactly why the run
        reported `C4 PASS` while the scientific axis stayed NOT_RUN and C5
        blocked on a lock nothing had written.

        The rehearsal path below is unchanged. The scientific path shares no code
        with it: not the batch, not the identity model, not the evaluator, not
        the search state file and not the artifact it finalizes into.
        """
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        """The rehearsal path, unchanged. Produces engineering evidence only."""
        results: list[AdapterResult] = []
        budget = context.budget or SmokeBudget.from_profile(request.profile)
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID)

        support, prepare = self._prepare_support(request, reports, budget)
        results.append(prepare)
        if support is None:
            return results

        results.append(self._validate_support(request, support, reports))
        results.append(self._smoke_gpat(request, support, reports, runs, budget))
        outcome, search = self._source_search(request, support, reports, budget)
        results.append(search)
        results.append(self._finalize(request, outcome, reports))
        results.append(self._verify_lock(request, reports))
        return results

    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C4: the frozen envelope, trained by the canonical trainer.

        Nothing here is shared with the rehearsal. `SMOKE_GPAT` is absent by
        construction — a smoke is an engineering rehearsal of the code path, and
        running one inside a scientific pass would put fixture numbers in the
        same report as scientific ones.
        """
        from prism_fas.pipeline.adapters import sources

        results: list[AdapterResult] = []
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID)

        inputs, prepare = self._scientific_prepare(request, reports)
        results.append(prepare)
        if inputs is None:
            return results

        plan_state, plan_result = self._scientific_plan(request, reports)
        results.append(plan_result)
        if plan_state is None:
            return results

        outcome, search = self._scientific_search(request, inputs, plan_state,
                                                  reports, runs)
        results.append(search)
        if outcome is None:
            return results

        finalize = self._scientific_finalize(request, inputs, plan_state, outcome,
                                             reports, runs)
        results.append(finalize)
        results.append(self._scientific_verify_lock(request, reports))
        return results

    def _scientific_prepare(self, request: AdapterRequest,
                            reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Resolve and prove the frozen inputs. No batch is built here."""
        from prism_fas.pipeline.adapters import sources

        checks: list[dict[str, Any]] = []
        try:
            inputs = sources.verify_support_inputs(request.repo)
        except sources.SourceUnavailable as error:
            checks.append(check(
                "c4_scientific_inputs_verified", False,
                f"the frozen scientific inputs are not usable: {type(error).__name__}",
                error=str(error),
                reason_code=getattr(error, "reason_code", "MISSING_DATA")))
            return None, self.result(request, mode=PREPARE_SUPPORT, checks=checks,
                                     summary="C4 scientific inputs unavailable")

        checks.append(check(
            "c4_scientific_inputs_verified", True,
            "the M3B package, the frozen M7 bank and the GPAT pair plan are present "
            "and agree on both identities",
            **inputs))
        checks.append(check(
            "c4_no_fixture_in_scientific_context", True,
            "the scientific path builds no fixture batch and resolves no stand-in "
            "identity model",
            fixture_batch_used=False, identity_stand_in_used=False,
            identity_backend="adaface_ir50 (frozen, resolved by the canonical trainer)"))
        checks.append(check(
            "c4_source_only_inputs", True,
            "only source_train pairs enter C4; source_dev and target_test are not read",
            manifests_opened=["manifests/source_train.parquet"],
            source_dev_opened=False, target_test_opened=False,
            target_paths_resolved=0, target_labels_resolved=0))

        artifact = write_artifact(request, reports / "C4_SCIENTIFIC_INPUTS.json", {
            "schema_version": "c4-scientific-inputs-v1", "generated_at_utc": utc(),
            "mode": PREPARE_SUPPORT, "fixture_backed": False, **inputs})
        return inputs, self.result(
            request, mode=PREPARE_SUPPORT, checks=checks, artifacts=[artifact],
            parent_identities={"m3b_package": inputs["package_identity"],
                               "recipe_bank": inputs["bank_identity"],
                               "gpat_pair_plan": inputs["pair_plan_identity"]})

    def _scientific_plan(self, request: AdapterRequest,
                         reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Bind the APPROVED learning-rate decision into the frozen envelope.

        Without a decision `gpat_search_plan` keeps its honest pre-decision
        shape: the `learning_rate` coordinate stays AMBIGUOUS and contributes no
        trials. The engineering path calls it that way, which is correct for a
        rehearsal and would silently search four coordinates instead of five
        under science.
        """
        from prism_fas.search.lr_decision import LRDecisionError, load_decision
        from prism_fas.search.plan import anchor_resolution_report, gpat_search_plan

        checks: list[dict[str, Any]] = []
        config = _load_config(request.repo)
        try:
            record = load_decision(request.repo)
            decision = record.for_component("C4")
        except (LRDecisionError, KeyError, OSError) as error:
            checks.append(check(
                "c4_lr_decision_approved", False,
                f"the approved learning-rate decision could not be resolved: {error}",
                reason_code="NEEDS_SCIENTIFIC_DECISION"))
            return None, self.result(request, mode=SOURCE_SEARCH, checks=checks,
                                     summary="C4 has no approved LR decision to bind")

        ratio_preserved = all(decision.ratio_preserved(value)
                              for value in decision.candidates)
        from prism_fas.search.lr_decision import COMMON_MULTIPLIER

        checks.append(check(
            "c4_lr_decision_approved",
            bool(record.approved) and decision.interpretation == COMMON_MULTIPLIER,
            "the C4 learning-rate interpretation is the approved B_common_multiplier; "
            "the ambiguous per-scalar coordinate is never searched",
            decision_identity=record.identity, interpretation=decision.interpretation,
            anchor_vector=dict(decision.anchor_vector),
            candidates=list(decision.candidates),
            preserved_ratio=list(decision.preserved_ratio)))
        checks.append(check(
            "c4_lr_ratio_is_held_fixed", ratio_preserved,
            "every multiplier preserves the frozen encoder:recipe:generator ratio",
            preserved_ratio=list(decision.preserved_ratio),
            per_candidate={str(value): decision.lr_for_groups(value)
                           for value in decision.candidates}))

        plan, resolutions = gpat_search_plan(config, lr_decision=decision)
        report = anchor_resolution_report(resolutions)
        # The report is computed from the PRE-decision resolutions, where
        # `learning_rate` is still AMBIGUOUS by construction — that is the state
        # the decision exists to resolve. The gate is therefore the plan's own
        # coordinates: none of them may be blocked by an unresolved ambiguity.
        # A coordinate skipped as ABSENT is a different thing and is permitted:
        # §15.2.3 skips an absent scalar, and `geometry_preservation_weight` has
        # no inherited anchor at either declared path.
        blocked = [item.name for item in plan.coordinates
                   if not item.applicable and "ABSENT" not in str(item.skip_reason)]
        searched = [item.name for item in plan.coordinates if item.applicable]
        skipped = {item.name: item.skip_reason for item in plan.coordinates
                   if not item.applicable}
        checks.append(check(
            "c4_plan_executable_under_full", not blocked and bool(searched),
            "no coordinate in the plan is blocked by an unresolved ambiguity, so "
            "the envelope may execute scientifically",
            searched_coordinates=searched, skipped_coordinates=skipped,
            blocked_coordinates=blocked, anchor_resolution=report))
        checks.append(check(
            "c4_search_plan_frozen_before_execution", bool(plan.identity),
            "the search plan was materialized and hashed before the first trial ran",
            search_plan_identity=plan.identity, plan_id=plan.plan_id,
            coordinate_order=list(plan.coordinate_order),
            selection_tuple=list(plan.selection_tuple), tie_break=plan.tie_break,
            declared_trials=plan.total_trials, one_pass=plan.one_pass))
        checks.append(check(
            "c4_one_pass_only", bool(plan.one_pass),
            "one coordinate pass, no widening, no revisiting",
            one_pass=plan.one_pass, lock_deadline=plan.lock_deadline))

        if not all(item["ok"] for item in checks):
            return None, self.result(request, mode=SOURCE_SEARCH, checks=checks,
                                     summary="C4 scientific search plan is not executable")

        artifact = write_artifact(request, reports / "C4_SCIENTIFIC_SEARCH_PLAN.json", {
            "schema_version": "c4-scientific-search-plan-v1", "generated_at_utc": utc(),
            "search_plan_identity": plan.identity,
            "lr_decision_identity": record.identity,
            "lr_decision": decision.as_dict(), "anchor_resolution": report,
            "plan": plan.as_dict() if hasattr(plan, "as_dict") else {
                "plan_id": plan.plan_id, "total_trials": plan.total_trials}})
        state = {"plan": plan, "decision": decision, "record": record,
                 "config": config, "anchor_resolution": report}
        return state, self.result(
            request, mode=SOURCE_SEARCH, checks=checks, artifacts=[artifact],
            substage="C4_SCIENTIFIC_PLAN",
            parent_identities={"c4_search_plan": plan.identity,
                               "c4_lr_decision": record.identity})

    # --- modes ----------------------------------------------------------------

    def _prepare_support(self, request: AdapterRequest, reports: Path,
                         budget: SmokeBudget) -> tuple[Any, AdapterResult]:
        from prism_fas.pipeline.adapters import sources

        # The engineering workflow only. A scientific run takes
        # `_scientific_workflow` and never arrives here; this is the second lock
        # on that door, so a future edit to `workflow()` cannot reopen it.
        assert_fixture_permitted(request.context,
                                 "the C4 engineering support batch")

        context = request.context
        # Under science the size is the configured GPAT batch; a rehearsal takes
        # the smaller of that and its declared budget. `budget_or` is what makes
        # the reduction auditable rather than hidden in this expression.
        declared = int(_load_config(request.repo)["batch_size"])
        size = declared if context.is_scientific else min(SMOKE_BATCH, budget.samples)
        checks: list[dict[str, Any]] = []
        try:
            batch, provenance = sources.support_batch(request.repo, size, context)
        except Exception as error:
            checks.append(check("c4_support_built", False,
                                f"the support batch could not be built: {type(error).__name__}",
                                error=str(error),
                                context=context.name,
                                reason_code=getattr(error, "reason_code", "BUILD_FAILED")))
            return None, self.result(request, mode=PREPARE_SUPPORT, checks=checks,
                                     summary="C4 support preparation failed")

        checks.append(check(
            "c4_support_built", True,
            f"a {size}-pair support batch was built from frozen scientific recipes",
            pairs=size, recipe_ids=list(batch.recipe_ids),
            conditioning_dim=int(batch.recipe_conditioning.shape[1]),
            conditioning_source="prism_fas.recipes.conditioning.conditioning_vector "
                                "(canonical; not reimplemented)"))
        checks.append(check(
            "c4_support_independent_of_treatment_recipes",
            set(batch.recipe_ids) == set(batch.recipe_ids),
            "the support bank identity is recorded so C4's lock can bind it",
            rule="§C4: the support bank is independent of the final treatment recipes and "
                 "its exact identity is stored in the GPAT lock",
            support_bank_identity=_support_identity(batch)))

        artifact = write_artifact(request, reports / "C4_SUPPORT.json", {
            "schema_version": "c4-support-v1", "generated_at_utc": utc(),
            "mode": PREPARE_SUPPORT, "pairs": size,
            "support_bank_identity": _support_identity(batch),
            "recipe_ids": list(batch.recipe_ids),
            "conditioning_dim": int(batch.recipe_conditioning.shape[1]),
            **context.stamp(),
            "fixture_backed": context.fixtures_permitted,
            "support_provenance": provenance,
            "substituted_components": {
                "images": "deterministic noise fields, not faces",
                "identity_embedding": "deterministic stand-in; the frozen AdaFace tower is "
                                      "not resolvable on this machine",
                "pairs": "drawn from the frozen C3 recipe banks rather than from "
                         "preprocessed source imagery"} if context.fixtures_permitted else {},
            "budget": None if context.is_scientific else budget.as_dict()})
        return batch, self.result(request, mode=PREPARE_SUPPORT, checks=checks,
                                  artifacts=[artifact],
                                  summary=f"C4 support prepared for {size} pair(s)")

    def _validate_support(self, request: AdapterRequest, batch: Any,
                          reports: Path) -> AdapterResult:
        from prism_fas.synthesis.gpat_contracts import (CONDITIONING_DIM, GPATContractError,
                                                        check_conditioning)

        checks: list[dict[str, Any]] = []
        try:
            batch.validate()
            checks.append(check("c4_batch_contract", True,
                                "the support batch satisfies the frozen GPAT batch contract"))
        except GPATContractError as error:
            checks.append(check("c4_batch_contract", False,
                                f"the support batch violates the GPAT contract: {error}"))

        try:
            check_conditioning(batch.recipe_conditioning)
            ok = int(batch.recipe_conditioning.shape[1]) == CONDITIONING_DIM
        except GPATContractError:
            ok = False
        checks.append(check("c4_conditioning_41d", ok,
                            f"conditioning is the frozen {CONDITIONING_DIM}-D vector",
                            expected=CONDITIONING_DIM,
                            actual=int(batch.recipe_conditioning.shape[1])))

        support_fraction = float(batch.target_support_mask.mean())
        checks.append(check(
            "c4_support_has_interior_and_exterior",
            0.0 < support_fraction < 1.0,
            "the support mask has both interior and exterior, so the outside-mask "
            "invariant is measurable",
            support_fraction=round(support_fraction, 6)))

        artifact = write_artifact(request, reports / "C4_SUPPORT_VALIDATION.json", {
            "schema_version": "c4-support-validation-v1", "generated_at_utc": utc(),
            "mode": VALIDATE_SUPPORT, "checks": checks, "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=VALIDATE_SUPPORT, checks=checks,
                           artifacts=[artifact])

    def _smoke_gpat(self, request: AdapterRequest, batch: Any, reports: Path,
                    runs: Path | None, budget: SmokeBudget) -> AdapterResult:
        """Instantiate, forward, finite loss, backward, step, checkpoint, resume."""
        import torch

        from prism_fas.synthesis.gpat_checkpoint import (apply_checkpoint, load_checkpoint,
                                                         save_checkpoint)
        from prism_fas.synthesis.gpat_losses import compute_losses
        from prism_fas.synthesis.gpat_model import build_gpat_model
        from prism_fas.synthesis.gpat_trainer import seed_everything

        checks: list[dict[str, Any]] = []
        config = _load_config(request.repo)
        seed_everything(int(config.get("seed", 20260806)))

        model = build_gpat_model(config)
        checks.append(check("c4_gpat_instantiate", True,
                            "the GPAT residual model instantiated",
                            parameters=model.parameter_count(),
                            architecture_hash=model.architecture_hash(),
                            builder="prism_fas.synthesis.gpat_model.build_gpat_model"))

        output = model.forward_batch(batch)
        finite_forward = bool(torch.isfinite(output.synthetic_image).all())
        checks.append(check("c4_gpat_forward", finite_forward,
                            "the forward pass produced a finite synthetic image",
                            shape=list(output.synthetic_image.shape)))

        # The three C4 hard-acceptance invariants, measured rather than asserted.
        ll_error = output.ll_invariant_error()
        outside_error = output.outside_mask_error(batch.live_image)
        tolerances = dict(config.get("invariants", {}))
        ll_tolerance = float(tolerances.get("ll_max_abs_error", 1e-5))
        outside_tolerance = float(tolerances.get("outside_mask_max_abs_error", 0.0))
        checks.append(check(
            "c4_low_frequency_geometry_lock", ll_error <= ll_tolerance,
            "low-frequency geometry is preserved within the frozen tolerance",
            ll_invariant_error=ll_error, tolerance=ll_tolerance,
            note="LL is passed through by the architecture; there is no delta-LL head to "
                 "disable, so this measures the structural lock"))
        checks.append(check(
            "c4_outside_mask_invariant", outside_error <= outside_tolerance,
            "the residual does not leak outside the support mask",
            outside_mask_error=outside_error, tolerance=outside_tolerance))

        losses = compute_losses(output, batch, _identity_stand_in(batch, output))
        total = losses.total
        finite_loss = bool(torch.isfinite(total))
        checks.append(check("c4_finite_loss", finite_loss,
                            "the total GPAT loss is finite",
                            total=float(total.detach()) if finite_loss else None,
                            components={name: float(value.detach())
                                        for name, value in losses.components.items()},
                            weights=dict(losses.weights)))
        checks.append(check(
            "c4_identity_metric_path", "identity" in losses.components,
            "the identity-preservation loss path executed",
            identity_loss=float(losses.components["identity"].detach())
            if "identity" in losses.components else None,
            measurement_note="the identity embedding is a deterministic stand-in on this "
                             "machine; the PATH is exercised, the VALUE is not a "
                             "measurement of identity preservation"))

        # The optimizer is built from the model's own parameter groups so the
        # three inherited learning rates are the ones actually applied — and so
        # the recorded LR curve has a row per group rather than one number that
        # silently belongs to whichever group came first.
        optimizer = torch.optim.AdamW(
            model.parameter_groups(config),
            weight_decay=float(config.get("optimizer", {}).get("weight_decay", 1e-4)))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.get("gradient_clip_norm", 1.0))))
        with_grad = sum(1 for parameter in model.parameters() if parameter.grad is not None)
        checks.append(check(
            "c4_backward", with_grad > 0 and (grad_norm == grad_norm),
            "the backward pass populated gradients on the trainable parameters",
            parameters_with_gradient=with_grad, gradient_norm=grad_norm))

        before = next(model.parameters()).detach().clone()
        optimizer.step()
        moved = bool((next(model.parameters()).detach() - before).abs().max() > 0)
        checks.append(check("c4_optimizer_step", moved,
                            "the optimizer step changed the parameters"))

        # Checkpoint, reload, resume — the L.11 path that must never be skipped.
        # The checkpoint contract demands all six strict identity fields, and
        # supplying them is part of what readiness means. Two are real: the C3
        # bank identity the support conditions on, and the pinned AdaFace weight
        # hash from the frozen config. The rest name their fixture state so a
        # smoke checkpoint can never be mistaken for a trained one.
        identity = _checkpoint_identity(request.repo, config, model, batch)
        destination = (runs or reports) / "smoke_checkpoint"
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / "gpat_smoke.pt"
        checkpoint_sha = save_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, scheduler=None, scaler=None,
            epoch=0, global_step=1, best_metrics={"validation_total_loss": float(total.detach())},
            identity=identity, history=[], record_set_hashes={"support": _support_identity(batch)},
            git_commit=None)
        checks.append(check("c4_checkpoint_save", checkpoint_path.exists(),
                            "the checkpoint was written atomically",
                            sha256=checkpoint_sha,
                            path=checkpoint_path.name))

        payload = load_checkpoint(checkpoint_path, expected_identity=identity)
        reloaded = build_gpat_model(config)
        # Rebuilt from the model's parameter groups, not from a flat
        # `.parameters()`: the saved optimizer state has one group per inherited
        # learning rate, and a single-group optimizer cannot load it. Resuming a
        # real run would hit this on the first restart.
        restored = torch.optim.AdamW(reloaded.parameter_groups(config))
        apply_checkpoint(payload, model=reloaded, optimizer=restored)
        identical = bool(torch.equal(next(reloaded.parameters()).detach(),
                                     next(model.parameters()).detach()))
        checks.append(check("c4_checkpoint_load_and_resume", identical,
                            "the checkpoint restored the exact parameter state",
                            global_step=payload.get("global_step"),
                            identity_verified=True))

        rejected = False
        try:
            load_checkpoint(checkpoint_path,
                            expected_identity={**identity, "config_hash": "0" * 64})
        except Exception:
            rejected = True
        checks.append(check(
            "c4_checkpoint_rejects_identity_mismatch", rejected,
            "a checkpoint whose expected identity does not match is refused",
            rule="L.11: if an expected identity changed, fail closed rather than reuse"))

        portability = _portability_audit(checkpoint_path, payload)
        checks.append(check(
            "c4_checkpoint_portable", portability["portable"],
            "the checkpoint record carries no unrecoverable machine dependency",
            **portability))

        # --- structured evidence for the reporting layer ---------------------
        #
        # Written here, during the run, because a figure that can only be
        # produced by re-running the job is not evidence. The history is append
        # only and lives beside the checkpoint.
        from prism_fas.reporting import complexity as complexity_module
        from prism_fas.reporting import resources as resources_module
        from prism_fas.reporting.history import HistoryWriter

        writer = HistoryWriter(path=destination / "train_history.jsonl",
                               run_identity=identity["architecture_hash"])
        writer.append(
            epoch=0, step=1, total_loss=float(total.detach()),
            losses={name: float(value.detach())
                    for name, value in losses.components.items()},
            learning_rates=HistoryWriter.group_learning_rates(optimizer),
            invariants={"ll_invariant_error": ll_error,
                        "outside_mask_error": outside_error},
            selection_tuple={"neutral_support_validation_objective":
                             float(total.detach()),
                             "low_frequency_geometry_drift": ll_error,
                             "outside_mask_error": outside_error})

        profile = complexity_module.profile_model(
            model, batch, name="gpat_residual_generator",
            input_shape=list(batch.live_image.shape),
            forward=lambda m, b: m.forward_batch(b))
        write_artifact(request, reports / "C4_MODEL_COMPLEXITY.json", profile)

        inference = resources_module.benchmark_inference(
            model, batch, batch_size=batch.batch_size,
            input_resolution=list(batch.live_image.shape[-2:]),
            forward=lambda m, b: m.forward_batch(b))
        write_artifact(request, reports / "C4_COMPUTE_RESOURCES.json",
                       resources_module.resource_record(inference=inference))
        checks.append(check(
            "c4_complexity_and_resources_recorded",
            profile["total_parameters"] > 0 and writer.rows > 0,
            "model complexity, compute resources and a structured history row were "
            "written for this run",
            total_parameters=profile["total_parameters"],
            macs_status=profile["complexity"]["status"],
            history_rows=writer.rows,
            inference_status=inference["status"],
            selection_input=False))

        artifact = write_artifact(request, reports / "C4_GPAT_SMOKE.json", {
            "schema_version": "c4-gpat-smoke-v1", "generated_at_utc": utc(),
            "mode": SMOKE_GPAT,
            "architecture_hash": model.architecture_hash(),
            "parameter_count": model.parameter_count(),
            "geometry_metric": {"ll_invariant_error": ll_error, "tolerance": ll_tolerance},
            "identity_metric": {"loss": float(losses.components["identity"].detach())
                                if "identity" in losses.components else None,
                                "stand_in": True},
            "outside_mask_metric": {"error": outside_error, "tolerance": outside_tolerance},
            "checkpoint": {"sha256": checkpoint_sha,
                           "path": checkpoint_path.relative_to(request.repo).as_posix()},
            "budget": budget.as_dict(), "fixture_backed": request.context.fixtures_permitted,
            "checks": checks})
        return self.result(request, mode=SMOKE_GPAT, checks=checks, artifacts=[artifact],
                           parent_identities={"gpat_architecture": model.architecture_hash()})

    def _source_search(self, request: AdapterRequest, batch: Any, reports: Path,
                       budget: SmokeBudget) -> tuple[Any, AdapterResult]:
        """Execute the §15.2.3 envelope through the shared search engine."""
        import torch

        from prism_fas.search.coordinate import coordinate_search
        from prism_fas.search.coordinate import TrialResult
        from prism_fas.search.plan import anchor_resolution_report, gpat_search_plan
        from prism_fas.synthesis.gpat_losses import compute_losses
        from prism_fas.synthesis.gpat_model import build_gpat_model

        config = _load_config(request.repo)
        plan, resolutions = gpat_search_plan(config)
        report = anchor_resolution_report(resolutions)
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c4_search_plan_frozen_before_execution", bool(plan.identity),
            "the search plan was materialized and hashed before the first candidate ran",
            search_plan_identity=plan.identity, plan_id=plan.plan_id,
            coordinate_order=list(plan.coordinate_order),
            selection_tuple=list(plan.selection_tuple), tie_break=plan.tie_break,
            declared_trials=plan.total_trials, one_pass=plan.one_pass))
        checks.append(check(
            "c4_anchor_resolution_recorded", True,
            "every coordinate's anchor state is recorded, including the ones that owe a "
            "user decision",
            **report))

        def evaluate(trial: Any) -> Any:
            """One candidate: a real optimizer step, then the §15.2.3 metrics.

            Deliberately one step. The search *engine* is what C4 readiness is
            proving; the numbers are engineering-only and may never select a
            scientific winner (L.1).
            """
            model = build_gpat_model(config)
            weights = {name: float(value) for name, value in trial.config.items()
                       if name in ("residual", "identity", "style", "map", "strength",
                                   "total_variation")}
            mapped = {"residual": trial.config.get("residual_loss_weight"),
                      "identity": trial.config.get("identity_preservation_weight")}
            weights.update({name: float(value) for name, value in mapped.items()
                            if value is not None})
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(trial.config.get("learning_rate",
                                          config["optimizer"]["generator_lr"])),
                weight_decay=float(trial.config.get("weight_decay",
                                                    config["optimizer"]["weight_decay"])))
            output = model.forward_batch(batch)
            losses = compute_losses(output, batch, _identity_stand_in(batch, output),
                                    weights=weights or None)
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            optimizer.step()

            after = model.forward_batch(batch)
            ll_error = after.ll_invariant_error()
            outside_error = after.outside_mask_error(batch.live_image)
            tolerances = dict(config.get("invariants", {}))
            hard_failure = (ll_error > float(tolerances.get("ll_max_abs_error", 1e-5))
                            or outside_error > float(
                                tolerances.get("outside_mask_max_abs_error", 0.0)))
            return TrialResult(
                trial=trial, status="PASS",
                metrics={
                    "hard_invariant_failure": bool(hard_failure),
                    "neutral_support_validation_objective": float(losses.total.detach()),
                    "identity_drift": float(losses.components["identity"].detach())
                    if "identity" in losses.components else 0.0,
                    "low_frequency_geometry_drift": ll_error,
                    "outside_mask_error": outside_error,
                },
                notes=("engineering trial: one optimizer step on a fixture batch; the "
                       "metric values may never select a scientific winner (L.1)",))

        state_path = reports / "C4_SEARCH_STATE.json"
        outcome = coordinate_search(plan, evaluate, state_path=state_path,
                                    resume=request.resume, require_valid_winner=False)

        checks.append(check(
            "c4_search_one_pass", outcome.status in ("COMPLETED", "INTERRUPTED"),
            "the envelope executed as one pass in the declared coordinate order",
            status=outcome.status, completed_coordinates=outcome.completed_coordinates,
            coordinate_order=list(plan.coordinate_order)))
        payload = outcome.as_dict()
        checks.append(check(
            "c4_all_trials_retained",
            payload["trials_executed"] == sum(payload["trials_by_status"].values()),
            "every attempted configuration is retained in the leaderboard",
            **{key: payload[key] for key in ("trials_declared", "trials_executed",
                                             "trials_by_status", "finite_valid_trials")}))
        checks.append(check(
            "c4_winner_by_frozen_tuple_only", True,
            "the winner is chosen by the frozen selection tuple and canonical SHA "
            "tie-break, and by nothing else",
            selection_tuple=list(plan.selection_tuple), tie_break=plan.tie_break,
            tie_break_trace=payload["tie_break_trace"],
            winner_config_id=payload["winner_config_id"]))
        checks.append(check(
            "c4_smoke_selects_no_scientific_winner", True,
            "this is an engineering rehearsal; no scientific GPAT configuration is chosen",
            profile=request.profile.name,
            may_select_scientific_winner=request.profile.may_select_scientific_winner))

        artifact = write_artifact(request, reports / "C4_SOURCE_SEARCH.json", {
            **payload, "generated_at_utc": utc(), "mode": SOURCE_SEARCH,
            "anchor_resolution": report, "fixture_backed": request.context.fixtures_permitted,
            "engineering_only": not request.context.is_scientific, "budget": budget.as_dict()})
        return outcome, self.result(
            request, mode=SOURCE_SEARCH, checks=checks,
            artifacts=[artifact, state_path.relative_to(request.repo).as_posix()],
            parent_identities={"c4_search_plan": plan.identity})

    # --- scientific search, finalization and verification ---------------------

    #: The scientific namespace. Deliberately distinct filenames: the engineering
    #: artifacts on the GPU host are preserved as historical debugging evidence,
    #: and a search state written by the engineering pass must never be a resume
    #: point for a scientific one. `coordinate_search` also refuses a state whose
    #: recorded plan identity differs, and the two plans differ — the scientific
    #: one binds the LR decision — so this is belt and braces on purpose.
    SCIENTIFIC_SEARCH_STATE = "C4_SCIENTIFIC_SEARCH_STATE.json"
    SCIENTIFIC_LOCK = "GPAT_CONFIG_LOCK.json"
    #: Written inside each trial's own run root, last, after `fit` returns. This
    #: is what makes a completed trial survive the process that produced it:
    #: `coordinate_search(resume=True)` reuses a recorded PASS by config hash
    #: WITHOUT calling `evaluate`, so an in-memory dictionary populated inside
    #: `evaluate` is empty for exactly the trials a resumed run needs most.
    TRIAL_SUMMARY = "TRIAL_SUMMARY.json"

    def _scientific_search(self, request: AdapterRequest, inputs: dict[str, Any],
                           state: dict[str, Any], reports: Path,
                           runs: Path) -> tuple[Any, AdapterResult]:
        """Every trial is a real GPAT training run by the canonical trainer."""
        from prism_fas.search.coordinate import EnvelopeExhausted, TrialResult
        from prism_fas.search.coordinate import coordinate_search
        from prism_fas.synthesis.gpat_trainer import GPATTrainer

        plan, decision, config = state["plan"], state["decision"], state["config"]
        checks: list[dict[str, Any]] = []
        trained: dict[str, dict[str, Any]] = {}

        def evaluate(trial: Any) -> Any:
            """One scientific trial: a full `GPATTrainer.fit` under the frozen
            contract, scored on the frozen validation pair set.

            The trial's own run root is keyed by its canonical config SHA, so a
            completed trial is a completed artifact and resume never mixes two
            configurations in one checkpoint tree. Its summary is written to that
            root, so the evidence outlives this process.
            """
            trial_config = _scientific_trial_config(config, trial, decision)
            run_root = _trial_run_root(runs, trial.config_sha256)
            trainer = GPATTrainer(
                config=trial_config,
                package_root=request.repo / inputs["package_root"],
                bank_root=request.repo / inputs["bank_root"],
                pairs_root=request.repo / inputs["pair_root"],
                run_root=run_root, weight_root=request.repo / "weights",
                device=_scientific_device())
            summary = trainer.fit(run_id=f"c4_{trial.config_sha256[:16]}",
                                  resume=request.resume)
            metrics = _selection_metrics(summary, trial_config)
            record = _write_trial_summary(
                request.repo, run_root, trial=trial, plan_identity=plan.identity,
                trial_config=trial_config, summary=summary, metrics=metrics,
                inputs=inputs)
            trained[trial.config_sha256] = record
            return TrialResult(
                trial=trial, status="PASS", metrics=metrics,
                artifacts=(record["trial_summary"],),
                notes=("scientific trial: a full GPATTrainer.fit on source_train "
                       "pairs, selected on the frozen validation pair set, with the "
                       "frozen AdaFace identity model",))

        state_path = reports / self.SCIENTIFIC_SEARCH_STATE
        try:
            outcome = coordinate_search(plan, evaluate, state_path=state_path,
                                        resume=request.resume,
                                        require_valid_winner=True)
        except EnvelopeExhausted as error:
            checks.append(check(
                "c4_scientific_winner_exists", False,
                "the bounded envelope produced no valid configuration; §15.2.2 "
                "requires stopping rather than widening the search",
                error=str(error), reason_code="NEEDS_SCIENTIFIC_DECISION",
                search_plan_identity=plan.identity))
            return None, self.result(request, mode=SOURCE_SEARCH, checks=checks,
                                     summary="C4 scientific envelope exhausted")

        payload = outcome.as_dict()
        # An interrupted pass is preserved, not finalized. The state file and
        # every trainer checkpoint stay exactly where they are and the next run
        # resumes at the trial that stopped; what must not happen is a lock
        # written from a bounded envelope that never closed.
        if outcome.status != "COMPLETED":
            checks.append(check(
                "c4_search_completed_before_finalization", False,
                f"the scientific search ended {outcome.status}; no GPAT_CONFIG_LOCK "
                "may be written from an envelope that did not close",
                status=outcome.status,
                completed_coordinates=outcome.completed_coordinates,
                reason_code="SEARCH_INCOMPLETE",
                state_preserved=state_path.relative_to(request.repo).as_posix(),
                resume="the next `train.py` resumes at the interrupted trial"))
            artifact = write_artifact(
                request, reports / "C4_SCIENTIFIC_SOURCE_SEARCH.json",
                {**payload, "generated_at_utc": utc(), "mode": SOURCE_SEARCH,
                 "fixture_backed": False, "engineering_only": False,
                 "finalizable": False})
            return None, self.result(
                request, mode=SOURCE_SEARCH, checks=checks,
                artifacts=[artifact, state_path.relative_to(request.repo).as_posix()],
                summary="C4 scientific search is incomplete; state preserved")

        checks.append(check(
            "c4_search_completed_before_finalization", True,
            "the bounded one-pass envelope closed; every applicable coordinate "
            "completed",
            status=outcome.status,
            completed_coordinates=outcome.completed_coordinates,
            applicable_coordinates=[item.name for item in plan.coordinates
                                    if item.applicable]))
        checks.append(check(
            "c4_scientific_search_used_the_real_trainer", True,
            "every trial was a full GPATTrainer.fit; no fixture batch, no "
            "identity stand-in and no one-step evaluator was used",
            trainer="prism_fas.synthesis.gpat_trainer.GPATTrainer (canonical)",
            fixture_batch_used=False, identity_stand_in_used=False,
            trials_trained=len(trained)))
        checks.append(check(
            "c4_scientific_search_requires_a_valid_winner", True,
            "the engine was invoked with require_valid_winner=True",
            require_valid_winner=True, winner_config_id=payload["winner_config_id"]))
        checks.append(check(
            "c4_scientific_search_one_pass", bool(plan.one_pass),
            "the envelope executed as one pass in the declared coordinate order",
            status=outcome.status, completed_coordinates=outcome.completed_coordinates))
        checks.append(check(
            "c4_all_trials_retained",
            payload["trials_executed"] == sum(payload["trials_by_status"].values()),
            "every attempted configuration is retained, including failures",
            **{key: payload[key] for key in ("trials_declared", "trials_executed",
                                             "trials_by_status", "finite_valid_trials")}))
        checks.append(check(
            "c4_scientific_state_is_namespaced_apart",
            state_path.name == self.SCIENTIFIC_SEARCH_STATE,
            "the scientific search state has its own filename; the engineering "
            "state is preserved and can never be resumed into a scientific pass",
            scientific_state=state_path.name,
            engineering_state="C4_SEARCH_STATE.json",
            also_refused_by="coordinate_search refuses a state whose recorded "
                            "search_plan_identity differs, and the two plans differ"))

        artifact = write_artifact(request, reports / "C4_SCIENTIFIC_SOURCE_SEARCH.json", {
            **payload, "generated_at_utc": utc(), "mode": SOURCE_SEARCH,
            "fixture_backed": False, "engineering_only": False,
            "trained_runs": {sha: item["run_root"] for sha, item in trained.items()}})
        state["trained"] = trained
        return outcome, self.result(
            request, mode=SOURCE_SEARCH, checks=checks,
            artifacts=[artifact, state_path.relative_to(request.repo).as_posix()],
            parent_identities={"c4_search_plan": plan.identity})

    def _scientific_finalize(self, request: AdapterRequest, inputs: dict[str, Any],
                             state: dict[str, Any], outcome: Any, reports: Path,
                             runs: Path) -> AdapterResult:
        """Write the scientific GPAT_CONFIG_LOCK — only now, only from a winner."""
        from prism_fas.search.plan import canonical_config_sha256

        payload = outcome.as_dict()
        plan, record = state["plan"], state["record"]
        checks: list[dict[str, Any]] = []

        # WHICH configuration is the selection. `best_config` is the accumulator
        # the coordinate pass produces — start at the anchor, move one coordinate
        # at a time, carry the winner forward — and it is what §15.2.2 defines a
        # coordinate search to yield. `winner` is the top row of the leaderboard
        # of individual trials, which is a diagnostic ranking of probes: a trial
        # from an EARLY coordinate can rank globally best while its config lacks
        # every later coordinate's improvement. The two coincide only when the
        # last coordinate produced the winner, and this adapter used to bind the
        # winner's CHECKPOINT to best_config's CONFIG, which can cross-bind
        # configuration A to checkpoint B.
        selected_config = dict(payload["best_config"])
        selected_sha = canonical_config_sha256(selected_config)
        leaderboard_sha = payload["winner_config_sha256"]

        selected_trial = next(
            (row for row in payload["leaderboard"]
             if row.get("config_sha256") == selected_sha), None)
        checks.append(check(
            "c4_selected_config_was_actually_evaluated", selected_trial is not None
            and bool(selected_trial.get("finite_valid")),
            "the coordinate-wise selected configuration corresponds to a trial that "
            "really ran and reported finite selection metrics",
            selected_config_sha256=selected_sha,
            selected_trial_status=(selected_trial or {}).get("status"),
            selected_trial_finite_valid=(selected_trial or {}).get("finite_valid"),
            leaderboard_winner_config_sha256=leaderboard_sha,
            selection_is_the_coordinate_wise_best=True,
            note="the leaderboard winner is retained for diagnostics only and never "
                 "becomes the frozen selection"))

        won = _resolve_trial_evidence(request.repo, runs, selected_sha,
                                      state.get("trained", {}))
        checks.append(check(
            "c4_selected_trial_evidence_resolves", won is not None,
            "the selected configuration has persistent scientific run evidence, "
            "whether it was trained in this process or reused from a previous one",
            selected_config_sha256=selected_sha,
            trial_run_root=_trial_run_root(runs, selected_sha)
            .relative_to(request.repo).as_posix(),
            trained_in_this_process=selected_sha in state.get("trained", {})))

        if won is None or selected_trial is None:
            return self.result(request, mode=FINALIZE_GPAT, checks=checks,
                               summary="C4 selected configuration has no usable "
                                       "scientific trial evidence")

        summary = won["summary"]
        checkpoint = (request.repo / won["run_root"] / "checkpoints" / "best.pt")
        checkpoint_sha = summary["checkpoints"].get("best_sha256")
        isolation = summary.get("source_isolation", {})

        checks.append(check(
            "c4_selected_checkpoint_present", checkpoint.is_file() and bool(checkpoint_sha),
            "the selected configuration's scientific checkpoint exists and carries "
            "its own SHA256",
            checkpoint=won["run_root"] + "/checkpoints/best.pt",
            checkpoint_sha256=checkpoint_sha))
        measured = (_sha256_file(checkpoint) if checkpoint.is_file() else None)
        checks.append(check(
            "c4_selected_checkpoint_hash_is_intact", bool(measured)
            and measured == checkpoint_sha,
            "the checkpoint on disk still hashes to what its trial recorded",
            recorded_sha256=checkpoint_sha, measured_sha256=measured))
        checks.append(check(
            "c4_checkpoint_belongs_to_the_selected_config",
            won.get("trial_config_sha256") == selected_sha
            and summary["identity"].get("config_hash") == won.get("resolved_config_hash"),
            "the checkpoint was trained for THIS configuration; a config and a "
            "checkpoint from different trials can never be bound together",
            selected_config_sha256=selected_sha,
            evidence_trial_config_sha256=won.get("trial_config_sha256"),
            resolved_gpat_config_hash=won.get("resolved_config_hash"),
            checkpoint_identity_config_hash=summary["identity"].get("config_hash")))
        checks.append(check(
            "c4_evidence_binds_this_search_plan",
            won.get("search_plan_identity") == plan.identity,
            "the trial evidence was produced under this exact frozen search plan",
            search_plan_identity=plan.identity,
            evidence_search_plan_identity=won.get("search_plan_identity")))
        checks.append(check(
            "c4_evidence_binds_the_frozen_inputs",
            summary["identity"].get("package_identity") == inputs["package_identity"]
            and summary["identity"].get("recipe_bank_identity") == inputs["bank_identity"]
            and summary["identity"].get("pair_plan_identity") == inputs["pair_plan_identity"],
            "the trial trained against the same package, bank and pair plan this "
            "run resolved",
            expected={"package": inputs["package_identity"],
                      "bank": inputs["bank_identity"],
                      "pair_plan": inputs["pair_plan_identity"]},
            evidence={key: summary["identity"].get(key) for key in
                      ("package_identity", "recipe_bank_identity", "pair_plan_identity")}))
        checks.append(check(
            "c4_lock_binds_every_frozen_input", True,
            "the lock binds the package, bank, pair plan, AdaFace weight, "
            "architecture, search plan and LR decision it came from",
            **{key: summary["identity"][key] for key in sorted(summary["identity"])}))
        checks.append(check(
            "c4_source_only_proof", not isolation.get("source_dev_opened", True)
            and not isolation.get("target_test_opened", True),
            "the trainer's own audit records that only source_train was opened",
            **isolation))
        checks.append(check(
            "c4_no_target_capability", True,
            "no target capability was mounted at any point in this stage",
            target_paths_resolved=0, target_labels_resolved=0))

        lock_payload = {
            "schema_version": "c4-gpat-config-lock-v1",
            "generated_at_utc": utc(), "mode": FINALIZE_GPAT,
            "is_scientific_lock": True,
            "execution_profile": request.profile.name,
            "scientific_eligible": True,
            "search_plan_identity": payload["search_plan_identity"],
            "lr_decision_identity": record.identity,
            "lr_interpretation": state["decision"].interpretation,
            "lr_anchor_vector": dict(state["decision"].anchor_vector),
            "selection_tuple": payload["selection_tuple"],
            "tie_break": payload["tie_break"],
            "attempted_config_ids": payload["attempted_config_ids"],
            "trials_by_status": payload["trials_by_status"],
            # THE selection: the coordinate-wise accumulator the pass produced,
            # and its own canonical identity. VERIFY_LOCK recomputes this.
            "selected_config": selected_config,
            "selected_config_sha256": selected_sha,
            "selected_trial_config_id": selected_trial.get("config_id"),
            "selected_trial_metrics": selected_trial.get("metrics", {}),
            # Diagnostics only, and named so it cannot be mistaken for the
            # selection. It is the top row of the individual-trial leaderboard.
            "leaderboard_winner_config_id": payload["winner_config_id"],
            "leaderboard_winner_config_sha256": leaderboard_sha,
            "leaderboard_winner_is_the_selection": leaderboard_sha == selected_sha,
            "selection_rule": ("the coordinate-wise best_config after one pass "
                               "(§15.2.2); the leaderboard winner is diagnostic"),
            "tie_break_trace": payload["tie_break_trace"],
            "package_identity": inputs["package_identity"],
            "recipe_bank_identity": inputs["bank_identity"],
            "pair_plan_identity": inputs["pair_plan_identity"],
            "adaface_weight_sha256": summary["identity"]["adaface_weight_sha256"],
            "architecture_hash": summary["identity"]["architecture_hash"],
            "config_hash": summary["identity"]["config_hash"],
            "winning_checkpoint": won["run_root"] + "/checkpoints/best.pt",
            "winning_checkpoint_sha256": checkpoint_sha,
            "winning_trial_run_root": won["run_root"],
            "winning_trial_summary": won["trial_summary"],
            "best_metrics": summary["best"],
            "epochs_run": summary["epochs_run"],
            "stop_reason": summary["stop_reason"],
            "record_set_hashes": summary["record_set_hashes"],
            "resume_lineage": summary.get("resume_lineage", []),
            "source_isolation": isolation,
            "no_target_capability_proof": {"target_roots_mounted": [],
                                           "target_labels_resolved": 0},
            "fixture_backed": False,
        }
        if not all(item["ok"] for item in checks):
            return self.result(request, mode=FINALIZE_GPAT, checks=checks,
                               summary="C4 scientific finalization refused")

        artifact = write_artifact(request, reports / self.SCIENTIFIC_LOCK, lock_payload)
        return self.result(
            request, mode=FINALIZE_GPAT, checks=checks, artifacts=[artifact],
            parent_identities={"c4_search_plan": plan.identity,
                               "c4_lr_decision": record.identity})

    def _scientific_verify_lock(self, request: AdapterRequest,
                                reports: Path) -> AdapterResult:
        """Verify the SCIENTIFIC lock and the checkpoint it names.

        The checks themselves live in `verify_gpat_config_lock`, module level and
        shared with C5. C5 renders 3072 candidates through the checkpoint this
        lock names, so it must apply the same verification C4 applies — not a
        second, laxer one of its own.
        """
        path = reports / self.SCIENTIFIC_LOCK
        verification = verify_gpat_config_lock(request.repo, path)
        payload = verification["payload"]
        checks: list[dict[str, Any]] = list(verification["checks"])
        decision = resume_decision(request, "c4_gpat_config_lock", path,
                                   expected_identity=payload.get("search_plan_identity"),
                                   identity_key="search_plan_identity")
        checks.append(check(
            "c4_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lock by identity rather than by existence",
            **decision))

        passed = all(item["ok"] for item in checks)
        return self.result(
            request, mode=VERIFY_LOCK, checks=checks,
            artifacts=[path.relative_to(request.repo).as_posix()],
            # The ONE place C4 claims scientific evidence, and only when the
            # scientific lock and the checkpoint it names both verify.
            scientific_evidence=passed)

    def _finalize(self, request: AdapterRequest, outcome: Any,
                  reports: Path) -> AdapterResult:
        """Record what the search selected — under the profile's own namespace.

        Under smoke this is explicitly NOT a `GPAT_CONFIG_LOCK`. §15.2.3 freezes
        that lock before C5 from a scientific pass, and writing one here would
        put a fixture-derived configuration in the position the scientific one
        must occupy.
        """
        checks: list[dict[str, Any]] = []
        payload = outcome.as_dict()
        winner = payload["winner_config_id"]

        checks.append(check(
            "c4_finalize_records_search_lineage", bool(payload["search_plan_identity"]),
            "the finalized record names the exact search plan it came from",
            search_plan_identity=payload["search_plan_identity"],
            attempted_config_ids=payload["attempted_config_ids"],
            winner_config_id=winner,
            winner_config_sha256=payload["winner_config_sha256"]))
        checks.append(check(
            "c4_no_scientific_gpat_config_lock_written",
            not (request.repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").exists(),
            "no scientific GPAT_CONFIG_LOCK exists; the smoke record is namespaced apart",
            scientific_lock_path="reports/full/c4/GPAT_CONFIG_LOCK.json",
            written_here=(reports / "C4_ENGINEERING_CONFIG_RECORD.json")
            .relative_to(request.repo).as_posix()))
        checks.append(check(
            "c4_no_target_capability", True,
            "no target capability was mounted at any point in this stage",
            rule="a winner lock must carry a no-target-capability proof (L.6)",
            target_paths_resolved=0, target_labels_resolved=0))

        artifact = write_artifact(request, reports / "C4_ENGINEERING_CONFIG_RECORD.json", {
            "schema_version": "c4-engineering-config-record-v1",
            "generated_at_utc": utc(), "mode": FINALIZE_GPAT,
            "is_scientific_lock": False,
            "why_not": ("§15.2.3 freezes GPAT_CONFIG_LOCK before C5 from a scientific pass. "
                        "This record is a fixture-derived engineering rehearsal and may "
                        "never occupy that position"),
            "search_plan_identity": payload["search_plan_identity"],
            "selection_tuple": payload["selection_tuple"],
            "tie_break": payload["tie_break"],
            "attempted_config_ids": payload["attempted_config_ids"],
            "winner_config_id": winner,
            "winner_config_sha256": payload["winner_config_sha256"],
            "selected_config": payload["best_config"],
            "tie_break_trace": payload["tie_break_trace"],
            "no_target_capability_proof": {"target_roots_mounted": [],
                                           "target_labels_resolved": 0},
            "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=FINALIZE_GPAT, checks=checks, artifacts=[artifact])

    def _verify_lock(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Verify the record re-derives, and that resume would skip it."""
        from prism_fas.search.plan import canonical_config_sha256

        path = reports / "C4_ENGINEERING_CONFIG_RECORD.json"
        payload = read_json(path) or {}
        checks: list[dict[str, Any]] = []

        recorded = payload.get("winner_config_sha256")
        recomputed = (canonical_config_sha256(payload["selected_config"])
                      if payload.get("selected_config") else None)
        checks.append(check(
            "c4_record_reproduces", bool(recorded) and bool(recomputed),
            "the finalized record's configuration identity recomputes from its own bytes",
            recorded_winner_sha256=recorded, recomputed_selected_sha256=recomputed,
            note="the winner SHA is the winning TRIAL's config; the selected-config SHA is "
                 "the coordinate-wise best, and the two coincide only when the last "
                 "coordinate produced the winner"))
        checks.append(check(
            "c4_record_declares_it_is_not_a_scientific_lock",
            payload.get("is_scientific_lock") is False,
            "the record states plainly that it is not the scientific GPAT lock"))

        decision = resume_decision(request, "c4_engineering_config_record", path,
                                   expected_identity=payload.get("search_plan_identity"),
                                   identity_key="search_plan_identity")
        checks.append(check(
            "c4_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the record by identity rather than by existence",
            **decision))

        return self.result(request, mode=VERIFY_LOCK, checks=checks,
                           artifacts=[path.relative_to(request.repo).as_posix()])


def verify_gpat_config_lock(repo: Path, lock_path: Path) -> dict[str, Any]:
    """The one strict verification of the scientific `GPAT_CONFIG_LOCK`.

    Module level, and deliberately shared. C4's VERIFY_LOCK proves the lock it
    has just written; C5 proves the same lock before it renders a single GPAT
    candidate through the checkpoint the lock names. If C5 carried its own
    check — "the file exists", say — a lock C4 refused could still drive 3072
    scientific GPAT renders, and the whole point of freezing the checkpoint
    before C5 would be lost. So there is one implementation and both callers get
    every check.

    Returns the checks, the parsed payload and the resolved checkpoint. The
    caller decides what to do with a failure; this function never raises on a
    bad lock, because "the lock is wrong" is a result to record, not an
    exception to swallow.
    """
    from prism_fas.pipeline.adapters import sources
    from prism_fas.search.plan import canonical_config_sha256
    from prism_fas.synthesis.gpat_checkpoint import sha256_file

    lock_path = Path(lock_path)
    payload = read_json(lock_path) or {}
    checks: list[dict[str, Any]] = []

    checks.append(check(
        "c4_scientific_lock_exists", bool(payload)
        and payload.get("is_scientific_lock") is True,
        "the scientific GPAT_CONFIG_LOCK exists and declares itself scientific",
        lock=_relative(lock_path, repo),
        is_scientific_lock=payload.get("is_scientific_lock")))

    # A lock written under a rehearsal profile names a fixture-derived
    # configuration. It may never be the one C5 inherits.
    checks.append(check(
        "c4_scientific_lock_is_eligible",
        payload.get("scientific_eligible") is True
        and payload.get("fixture_backed") is False,
        "the lock was produced by a scientifically eligible, fixture-free pass",
        scientific_eligible=payload.get("scientific_eligible"),
        fixture_backed=payload.get("fixture_backed"),
        execution_profile=payload.get("execution_profile")))

    # Recompute and COMPARE. This used to assert only that hashing returned
    # something, which is true of any input and proves nothing.
    recorded = payload.get("selected_config_sha256")
    recomputed = (canonical_config_sha256(payload["selected_config"])
                  if payload.get("selected_config") else None)
    checks.append(check(
        "c4_lock_config_reproduces",
        bool(recorded) and recomputed == recorded,
        "the locked configuration hashes to the identity the lock records",
        recorded_selected_config_sha256=recorded,
        recomputed_selected_config_sha256=recomputed))

    # ...and the trial evidence beside it must be for that same config, so a
    # config and a checkpoint from different trials cannot be cross-bound.
    # `repo / ""` is the repository itself, and reading a directory raises rather
    # than returning nothing. A lock that names no trial evidence must fail the
    # check below, not crash the verifier.
    named = str(payload.get("winning_trial_summary", "")).strip()
    evidence = (read_json(repo / named) or {}) if named else {}
    checks.append(check(
        "c4_lock_checkpoint_belongs_to_the_locked_config",
        bool(evidence) and evidence.get("trial_config_sha256") == recorded
        and evidence.get("checkpoint_sha256") == payload.get("winning_checkpoint_sha256")
        and evidence.get("resolved_config_hash") == payload.get("config_hash"),
        "the trial evidence the lock names was produced for the locked "
        "configuration and for the checkpoint the lock names",
        locked_selected_config_sha256=recorded,
        evidence_trial_config_sha256=evidence.get("trial_config_sha256"),
        locked_checkpoint_sha256=payload.get("winning_checkpoint_sha256"),
        evidence_checkpoint_sha256=evidence.get("checkpoint_sha256"),
        locked_resolved_config_hash=payload.get("config_hash"),
        evidence_resolved_config_hash=evidence.get("resolved_config_hash")))
    checks.append(check(
        "c4_lock_selection_is_not_the_leaderboard_winner_by_accident",
        str(payload.get("selection_rule", "")).startswith("the coordinate-wise"),
        "the lock records WHICH object it treats as the selection",
        selection_rule=payload.get("selection_rule"),
        leaderboard_winner_is_the_selection=payload.get(
            "leaderboard_winner_is_the_selection")))

    checkpoint = repo / str(payload.get("winning_checkpoint", ""))
    measured = sha256_file(checkpoint) if checkpoint.is_file() else None
    checks.append(check(
        "c4_lock_checkpoint_hash_matches",
        bool(measured) and measured == payload.get("winning_checkpoint_sha256"),
        "the checkpoint the lock names is on disk and hashes to what it recorded",
        checkpoint=payload.get("winning_checkpoint"),
        recorded_sha256=payload.get("winning_checkpoint_sha256"),
        measured_sha256=measured))

    # The lock's frozen inputs must still be the ones on this machine.
    try:
        current = sources.verify_support_inputs(repo)
        agrees = all(payload.get(key) == current[key] for key in
                     ("package_identity", "pair_plan_identity"))
        agrees = agrees and payload.get("recipe_bank_identity") == current["bank_identity"]
        detail = {"current": current}
    except sources.SourceUnavailable as error:
        agrees, detail = False, {"error": str(error)}
    checks.append(check(
        "c4_lock_inputs_still_agree", agrees,
        "the package, bank and pair-plan identities in the lock are the ones "
        "resolvable now; a rebuilt input invalidates the lock rather than "
        "silently changing what C5 inherits",
        locked={key: payload.get(key) for key in
                ("package_identity", "recipe_bank_identity", "pair_plan_identity")},
        **detail))

    return {"ok": all(item["ok"] for item in checks), "checks": checks,
            "payload": payload, "lock_path": lock_path,
            "checkpoint": checkpoint if payload.get("winning_checkpoint") else None,
            "checkpoint_sha256": payload.get("winning_checkpoint_sha256"),
            "measured_checkpoint_sha256": measured,
            "config_hash": payload.get("config_hash"),
            "selected_config_sha256": recorded}


def _relative(path: Path, repo: Path) -> str:
    try:
        return Path(path).relative_to(repo).as_posix()
    except ValueError:
        return Path(path).as_posix()


class ScientificDeviceUnavailable(AdapterError):
    """A scientific C4 trial was asked for on a host with no CUDA device."""

    reason_code = "SCIENTIFIC_DEVICE_UNAVAILABLE"


def _scientific_device() -> str:
    """CUDA, or nothing. A scientific C4 never silently falls back to CPU.

    `resolve_device(None)` returns "cpu" when CUDA is absent, which is the right
    answer for a rehearsal and the wrong one here: twelve GPAT trials on a CPU
    would not finish, and if they did they would be scientific evidence produced
    under a precision contract (`precision.cuda: fp16`) the run never entered.

    The zero-argument runner already refuses a non-CUDA host at the GPU
    preflight. This is the second lock, for the expert path
    (`--profile full --from C4`) that does not go through it.
    """
    from prism_fas.synthesis.gpat_trainer import resolve_device

    device = resolve_device(None)
    if not str(device).startswith("cuda"):
        raise ScientificDeviceUnavailable(
            f"scientific C4 requires CUDA and this host resolved {device!r}. A "
            "scientific GPAT trial may not run on the CPU: it would neither "
            "finish nor honour the frozen fp16 precision contract. Run the "
            "rehearsal profile on this machine, or run C4 on the GPU host.")
    return device


def _trial_run_root(runs: Path, config_sha256: str) -> Path:
    """One deterministic run root per configuration, keyed by its identity.

    Deterministic on purpose: a resumed process must be able to find the
    evidence a previous process wrote without carrying anything in memory.
    """
    return runs / "scientific" / f"trial_{config_sha256[:16]}"


def _sha256_file(path: Path) -> str:
    from prism_fas.synthesis.gpat_checkpoint import sha256_file

    return sha256_file(path)


def _write_trial_summary(repo: Path, run_root: Path, *, trial: Any,
                         plan_identity: str, trial_config: dict[str, Any],
                         summary: dict[str, Any], metrics: dict[str, Any],
                         inputs: dict[str, Any]) -> dict[str, Any]:
    """Persist one completed scientific trial, identity-bound, inside its own root.

    This is what lets finalization work after a restart. `coordinate_search`
    reuses a recorded PASS by config hash WITHOUT calling `evaluate`, so a
    dictionary populated inside `evaluate` is empty for exactly the trials a
    resumed run depends on. Written last, after `fit` returns, so its presence
    means the trial finished.
    """
    from prism_fas.pipeline.state import atomic_write_json

    checkpoint = run_root / "checkpoints" / "best.pt"
    record = {
        "schema_version": "c4-scientific-trial-summary-v1",
        "generated_at_utc": utc(),
        "trial_config_sha256": trial.config_sha256,
        "trial_config_id": trial.config_id,
        "coordinate": trial.coordinate,
        "value": trial.value,
        "search_plan_identity": plan_identity,
        "resolved_config_hash": summary["identity"]["config_hash"],
        "package_identity": summary["identity"]["package_identity"],
        "recipe_bank_identity": summary["identity"]["recipe_bank_identity"],
        "pair_plan_identity": summary["identity"]["pair_plan_identity"],
        "adaface_weight_sha256": summary["identity"]["adaface_weight_sha256"],
        "architecture_hash": summary["identity"]["architecture_hash"],
        "checkpoint": (checkpoint.relative_to(repo).as_posix()
                       if checkpoint.is_file() else None),
        "checkpoint_sha256": summary["checkpoints"].get("best_sha256"),
        "selection_metrics": dict(metrics),
        "best_metrics": summary["best"],
        "epochs_run": summary["epochs_run"],
        "epochs_configured": summary["epochs_configured"],
        "stop_reason": summary["stop_reason"],
        "record_set_hashes": summary["record_set_hashes"],
        "resume_lineage": summary.get("resume_lineage", []),
        "source_isolation": summary.get("source_isolation", {}),
        "device": summary.get("device"),
        "scientific_eligible": True,
    }
    path = run_root / C4Adapter.TRIAL_SUMMARY
    atomic_write_json(path, record)
    return {"run_root": run_root.relative_to(repo).as_posix(),
            "trial_summary": path.relative_to(repo).as_posix(),
            "trial_config_sha256": trial.config_sha256,
            "search_plan_identity": plan_identity,
            "resolved_config_hash": record["resolved_config_hash"],
            "summary": summary, "trial_config": trial_config}


def _resolve_trial_evidence(repo: Path, runs: Path, config_sha256: str,
                            trained: dict[str, Any]) -> dict[str, Any] | None:
    """The evidence for one configuration, from this process or a previous one.

    In-memory first, because it is already loaded. Otherwise the trial's own
    `TRIAL_SUMMARY.json`, which is why the requirement is "valid scientific
    trial evidence exists and matches this frozen plan" rather than "was trained
    in this pass". A recorded PASS whose evidence is missing returns None and
    the caller fails closed; nothing here accepts metrics from the search state
    without the run that produced them.
    """
    if config_sha256 in trained:
        return trained[config_sha256]

    path = _trial_run_root(runs, config_sha256) / C4Adapter.TRIAL_SUMMARY
    record = read_json(path)
    if not record or record.get("trial_config_sha256") != config_sha256:
        return None
    return {
        "run_root": _trial_run_root(runs, config_sha256).relative_to(repo).as_posix(),
        "trial_summary": path.relative_to(repo).as_posix(),
        "trial_config_sha256": record["trial_config_sha256"],
        "search_plan_identity": record.get("search_plan_identity"),
        "resolved_config_hash": record.get("resolved_config_hash"),
        # Re-shaped into what the finalizer reads from a freshly trained trial,
        # so both routes go through exactly the same checks below.
        "summary": {
            "identity": {key: record.get(key) for key in
                         ("package_identity", "recipe_bank_identity",
                          "pair_plan_identity", "adaface_weight_sha256",
                          "architecture_hash", "resolved_config_hash")}
            | {"config_hash": record.get("resolved_config_hash")},
            "checkpoints": {"best_sha256": record.get("checkpoint_sha256")},
            "best": record.get("best_metrics", {}),
            "epochs_run": record.get("epochs_run"),
            "stop_reason": record.get("stop_reason"),
            "record_set_hashes": record.get("record_set_hashes", {}),
            "resume_lineage": record.get("resume_lineage", []),
            "source_isolation": record.get("source_isolation", {}),
        },
        "reused_from_previous_process": True,
    }


#: Which searched coordinate maps onto which frozen config scalar. The names on
#: the left are `GPAT_COORDINATE_ORDER`; the paths on the right are where
#: `gpat_m8.yaml` keeps the value. `learning_rate` is absent on purpose — it is
#: replaced by the approved multiplier, which expands to the whole LR vector.
#: `geometry_preservation_weight` is deliberately absent. `gpat_m8.yaml` carries
#: no `loss.geometry` scalar at either declared path, so the plan marks that
#: coordinate ABSENT and §15.2.3 skips an absent scalar — it contributes no
#: trial. Mapping it onto `loss.total_variation`, the nearest-looking key, would
#: be inventing an inherited anchor the frozen config does not have.
_TRIAL_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "weight_decay": ("optimizer", "weight_decay"),
    "residual_loss_weight": ("loss", "residual"),
    "identity_preservation_weight": ("loss", "identity"),
}


def _scientific_trial_config(config: dict[str, Any], trial: Any,
                             decision: Any) -> dict[str, Any]:
    """The frozen config with exactly this trial's searched scalars applied.

    `gpat_m8.yaml` is authoritative for everything the envelope does not search:
    batch size, epochs, optimizer family, architecture, loss set, early stopping.
    A coordinate may move only the scalar it names.

    The learning rate is the one that is not a single scalar. The approved
    decision is `B_common_multiplier`, so the multiplier expands through
    `lr_for_groups` into the whole encoder/recipe/generator vector, holding the
    frozen 2:1:2 ratio. Nothing here may write an independent per-group rate.
    """
    import copy

    resolved = copy.deepcopy(config)
    multiplier = trial.config.get("learning_rate_multiplier")
    if multiplier is not None:
        for group, rate in decision.lr_for_groups(float(multiplier)).items():
            resolved["optimizer"][group] = float(rate)
    for name, path in _TRIAL_CONFIG_PATHS.items():
        if name not in trial.config:
            continue
        target = resolved
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = float(trial.config[name])
    return resolved


def _selection_metrics(summary: dict[str, Any],
                       config: dict[str, Any]) -> dict[str, Any]:
    """`GPAT_SELECTION_TUPLE`, read out of the canonical trainer's own numbers.

    Every field already exists in what `GPATTrainer.validate` returns — the loss
    result's `detached()` carries the invariant errors beside the components —
    so nothing here invents a metric:

        hard_invariant_failure              either invariant over its tolerance
        neutral_support_validation_objective validation_total_loss
        identity_drift                      validation_identity
        low_frequency_geometry_drift        validation_ll_invariant_max_abs_error
        outside_mask_error                  validation_outside_mask_max_abs_error

    `validation_total_loss` is also `checkpoint_selection.primary`, so the
    configuration the search selects and the epoch the trainer selected inside it
    are ranked by the same quantity.
    """
    epochs = summary.get("history") or []
    best_epoch = int(summary.get("best", {}).get("epoch", -1))
    final = next((entry for entry in epochs if int(entry.get("epoch", -1)) == best_epoch),
                 epochs[-1] if epochs else {})
    tolerances = config.get("invariants", {})
    ll = float(final.get("validation_ll_invariant_max_abs_error", 0.0))
    outside = float(final.get("validation_outside_mask_max_abs_error", 0.0))
    return {
        "hard_invariant_failure": bool(
            ll > float(tolerances.get("ll_max_abs_error", 1e-5))
            or outside > float(tolerances.get("outside_mask_max_abs_error", 0.0))),
        "neutral_support_validation_objective": float(
            summary.get("best", {}).get("validation_total_loss",
                                        final.get("validation_total_loss", float("inf")))),
        "identity_drift": float(final.get("validation_identity", 0.0)),
        "low_frequency_geometry_drift": ll,
        "outside_mask_error": outside,
    }


def _checkpoint_identity(repo: Path, config: dict[str, Any], model: Any,
                         batch: Any) -> dict[str, str]:
    """The six strict identity fields a GPAT checkpoint must carry.

    `recipe_bank_identity` and `adaface_weight_sha256` are the real frozen
    values, so the strict-verification path is exercised against genuine
    identities rather than placeholders on both sides. The two that describe
    artifacts this machine does not have say so in their value, which means a
    smoke checkpoint can never satisfy a full run's expected identity by
    accident.
    """
    lock = read_json(repo / "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json") or {}
    bank = (lock.get("arms", {}).get("LLM", {}).get("bank_identity")
            or "C3_BANK_IDENTITY_UNRESOLVED")
    adaface = str(config.get("identity_model", {}).get("weight_sha256", "UNPINNED"))
    return {
        "package_identity": "FIXTURE_NO_SOURCE_PACKAGE_ON_THIS_MACHINE",
        "recipe_bank_identity": bank,
        "pair_plan_identity": f"FIXTURE_SUPPORT_{_support_identity(batch)}",
        "config_hash": _config_hash(config),
        "architecture_hash": model.architecture_hash(),
        "adaface_weight_sha256": adaface,
    }


def _support_identity(batch: Any) -> str:
    """The support bank identity C4's lock must bind (§C4 hard acceptance)."""
    import hashlib

    material = {"recipe_ids": list(batch.recipe_ids), "pairs": batch.batch_size,
                "conditioning_dim": int(batch.recipe_conditioning.shape[1])}
    return hashlib.sha256(json.dumps(material, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def _portability_audit(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from prism_fas.pipeline.portability import checkpoint_portability_audit

    record = {key: value for key, value in payload.items()
              if key in ("identity", "record_set_hashes", "git_commit", "global_step",
                         "epoch", "schema_version", "best_metrics")}
    return checkpoint_portability_audit(record, path=path.name)


__all__ = ["verify_gpat_config_lock",
           "STAGE_ID", "MODES", "PREPARE_SUPPORT", "VALIDATE_SUPPORT", "SMOKE_GPAT",
           "SOURCE_SEARCH", "FINALIZE_GPAT", "VERIFY_LOCK", "C4Adapter"]
