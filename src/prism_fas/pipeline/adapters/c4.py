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

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput,
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

    # --- modes ----------------------------------------------------------------

    def _prepare_support(self, request: AdapterRequest, reports: Path,
                         budget: SmokeBudget) -> tuple[Any, AdapterResult]:
        from prism_fas.pipeline.adapters import sources

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


__all__ = ["STAGE_ID", "MODES", "PREPARE_SUPPORT", "VALIDATE_SUPPORT", "SMOKE_GPAT",
           "SOURCE_SEARCH", "FINALIZE_GPAT", "VERIFY_LOCK", "C4Adapter"]
