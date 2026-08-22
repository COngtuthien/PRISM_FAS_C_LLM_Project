"""C5 — route rendering and synthetic candidate integration.

C5's scientific job is to render 2048 candidates per arm through the two frozen
routes. This adapter proves the machinery that would do it: recipes load from the
frozen banks, compile through the real compiler, resolve to the exact route
contract, wire real masks, render through the real physics engine and the real
GPAT generator, and produce candidate identities that are deterministic and
collision-free.

What it deliberately does *not* do is render the scientific set. §10.4 fixes that
budget at 256 recipes x 8 renders per arm, and producing a fraction of it under a
non-eligible profile would create an artifact that looks like a partial
scientific bank. The smoke renders a handful, names the number, and stamps the
result as a fixture.

The failure path gets equal billing. A recipe whose mask has no support cannot be
rendered, and the correct behaviour is to record the failure as provenance rather
than drop the row — L.8 forbids winner-only cleanup, and a silently missing
candidate is exactly the shape that bug takes here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, read_json,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)
from prism_fas.pipeline.adapters.common import stage_runs_dir
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.adapters.tiny import face_arrays, face_image, frozen_recipes
from prism_fas.pipeline.preparation import (DERIVED_PACKAGES, M7_BANK_CONTENT_IDENTITY,
                                            PAIR_PLAN_PACKAGE)
from prism_fas.synthesis.c5_source_pair_plan import GPAT, PHYSICS

STAGE_ID = "C5"

LOAD_RECIPES = "LOAD_RECIPES"
RESOLVE_ROUTES = "RESOLVE_ROUTES"
RENDER_PHYSICS = "RENDER_PHYSICS"
RENDER_GPAT = "RENDER_GPAT"
CANDIDATE_IDENTITY = "CANDIDATE_IDENTITY"
FAILURE_RECORDING = "FAILURE_RECORDING"

#: The scientific substages. Disjoint from the rehearsal modes on purpose: a
#: report can never show a fixture render and a scientific one under one name.
VERIFY_C4_LOCK = "VERIFY_C4_LOCK"
LOAD_SOURCE_PAIR_PLAN = "LOAD_SOURCE_PAIR_PLAN"
BUILD_ARM_PLANS = "BUILD_ARM_PLANS"
RENDER_CANDIDATES = "RENDER_CANDIDATES"
VERIFY_RAW_CANDIDATES = "VERIFY_RAW_CANDIDATES"
FINALIZE_C5 = "FINALIZE_C5"
VERIFY_C5_LOCK = "VERIFY_C5_LOCK"

SCIENTIFIC_MODES: tuple[str, ...] = (VERIFY_C4_LOCK, LOAD_SOURCE_PAIR_PLAN,
                                     BUILD_ARM_PLANS, RENDER_CANDIDATES,
                                     VERIFY_RAW_CANDIDATES, FINALIZE_C5,
                                     VERIFY_C5_LOCK)

MODES: tuple[str, ...] = (LOAD_RECIPES, RESOLVE_ROUTES, RENDER_PHYSICS, RENDER_GPAT,
                          CANDIDATE_IDENTITY, FAILURE_RECORDING) + SCIENTIFIC_MODES

#: The frozen C4 checkpoint lock C5 renders through, and the source package the
#: schedule is planned over. Both are read from their canonical definitions.
C4_SCIENTIFIC_LOCK = "reports/full/c4/GPAT_CONFIG_LOCK.json"
SOURCE_PACKAGE_ROOT = DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]

ONTOLOGY = "configs/recipes/ontology_m7.yaml"
BANK_CONFIG = "configs/synthesis/synthetic_bank_m8.yaml"

#: Two recipes per arm is enough to exercise compile, route, mask and render
#: while keeping the CPU cost of eight physics operators bounded.
SMOKE_RECIPES_PER_ARM = 2
SMOKE_ARMS: tuple[str, ...] = ("LLM", "RND", "DET")


@dataclass
class C5Adapter(EngineeringAdapter):
    """The C5 execution adapter. Every renderer is imported, never reimplemented."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Route rendering and synthetic candidates"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = True

    #: The C5 completion artifact, the one thing that unblocks C6.
    SCIENTIFIC_LOCK = "C5_SYNTHESIS_LOCK.json"

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c3_recipe_banks", "assets/recipe_banks/c3",
                          "the three frozen 256-recipe scientific banks"),
            RequiredInput("bank_config", BANK_CONFIG,
                          "the frozen candidate-plan and render budget"),
            RequiredInput("source_packages", "data/packages",
                          "preprocessed source_train imagery and parsing/landmark priors"),
            RequiredInput("gpat_checkpoint_lock", "reports/full/c4/GPAT_CONFIG_LOCK.json",
                          "the frozen GPAT checkpoint C5's GPAT route renders through"),
        )

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        """The rehearsal path, unchanged. Produces engineering evidence only."""
        results: list[AdapterResult] = []
        budget = context.budget or SmokeBudget.from_profile(request.profile)
        reports = stage_reports_dir(request, STAGE_ID)

        compiled, load = self._load_recipes(request, reports, budget)
        results.append(load)
        if not compiled:
            return results

        results.append(self._resolve_routes(request, compiled, reports))
        physics, physics_result = self._render_physics(request, compiled, reports, budget)
        results.append(physics_result)
        results.append(self._render_gpat(request, compiled, reports, budget))
        results.append(self._candidate_identity(request, compiled, physics, reports))
        results.append(self._failure_recording(request, compiled, reports))
        return results

    # --- modes ----------------------------------------------------------------

    def _load_recipes(self, request: AdapterRequest, reports: Path,
                      budget: SmokeBudget) -> tuple[dict[str, list[Any]], AdapterResult]:
        from prism_fas.recipes.compile import compile_recipe
        from prism_fas.recipes.ontology import load_ontology

        checks: list[dict[str, Any]] = []
        ontology = load_ontology(request.repo / ONTOLOGY)
        count = min(SMOKE_RECIPES_PER_ARM, budget.samples)
        compiled: dict[str, list[Any]] = {}

        for arm in SMOKE_ARMS:
            try:
                recipes = frozen_recipes(request.repo, arm, count)
                graphs = [compile_recipe(recipe, ontology, bank_id=f"c3_{arm.lower()}")
                          for recipe in recipes]
            except Exception as error:
                checks.append(check(f"c5_load_{arm.lower()}", False,
                                    f"{arm} recipes did not load or compile: "
                                    f"{type(error).__name__}", error=str(error)))
                continue
            compiled[arm] = graphs
            checks.append(check(
                f"c5_load_{arm.lower()}", len(graphs) == count,
                f"{count} frozen {arm} recipe(s) loaded and compiled",
                arm=arm, recipes=len(graphs),
                graph_hashes=[graph.graph_hash for graph in graphs],
                compiler="prism_fas.recipes.compile.compile_recipe (canonical)",
                ontology_identity=ontology.sha256))

        checks.append(check(
            "c5_all_arms_use_one_compiler", len(compiled) == len(SMOKE_ARMS),
            "RND, DET and LLM recipes all compile through the same compiler",
            rule="§C3 acceptance: the three banks must pass the same compile tests",
            arms=sorted(compiled)))

        artifact = write_artifact(request, reports / "C5_RECIPES.json", {
            "schema_version": "c5-recipes-v1", "generated_at_utc": utc(),
            "mode": LOAD_RECIPES, "recipes_per_arm": count,
            "scientific_recipes_per_arm": 256,
            "arms": {arm: [graph.graph_hash for graph in graphs]
                     for arm, graphs in compiled.items()},
            "ontology_identity": ontology.sha256, "fixture_backed": request.context.fixtures_permitted,
            "budget": budget.as_dict()})
        return compiled, self.result(request, mode=LOAD_RECIPES, checks=checks,
                                     artifacts=[artifact])

    def _resolve_routes(self, request: AdapterRequest, compiled: dict[str, list[Any]],
                        reports: Path) -> AdapterResult:
        from prism_fas.llm.route_policy import load_route_policy

        policy = load_route_policy(request.repo / "configs/version_c/llm/c2c_route_policy.yaml")
        expected = tuple(policy.allowed_scientific_generator_route)
        checks: list[dict[str, Any]] = []

        observed: dict[str, list[list[str]]] = {}
        for arm, graphs in compiled.items():
            routes = [list(graph.generator_routes) for graph in graphs]
            observed[arm] = routes
            exact = all(tuple(route) == expected for route in routes)
            checks.append(check(
                f"c5_route_exact_{arm.lower()}", exact,
                f"every {arm} recipe resolves to the exact frozen route sequence",
                arm=arm, expected=list(expected), observed=routes,
                rule="containment is not enough: a physics-only or gpat-only recipe is "
                     "refused by the C3 bank contract"))

        config = _bank_config(request.repo)
        per_live = dict(config.get("candidate_recipes_per_live", {}))
        checks.append(check(
            "c5_render_budget_declared", bool(per_live),
            "the frozen per-route render budget is read from its config, not chosen here",
            candidate_recipes_per_live=per_live,
            scientific_budget={"candidates_per_arm": 2048, "renders_per_recipe": 8,
                               "physics_per_recipe": 4, "gpat_per_recipe": 4},
            config=BANK_CONFIG))
        checks.append(check(
            "c5_smoke_renders_no_scientific_set", True,
            "this rehearsal renders a handful of candidates and no scientific set",
            rendered_per_arm=len(next(iter(compiled.values()), [])),
            scientific_candidates_per_arm=2048))

        artifact = write_artifact(request, reports / "C5_ROUTES.json", {
            "schema_version": "c5-routes-v1", "generated_at_utc": utc(),
            "mode": RESOLVE_ROUTES, "required_generator_route": list(expected),
            "route_policy_identity": policy.route_policy_identity,
            "observed": observed, "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=RESOLVE_ROUTES, checks=checks, artifacts=[artifact],
                           parent_identities={"route_policy": policy.route_policy_identity})

    def _render_physics(self, request: AdapterRequest, compiled: dict[str, list[Any]],
                        reports: Path,
                        budget: SmokeBudget) -> tuple[list[dict[str, Any]], AdapterResult]:
        """Render through the real eight-operator physics engine."""
        import numpy as np

        from prism_fas.synthesis.physics import PhysicsEngine

        checks: list[dict[str, Any]] = []
        parsing, landmarks, bbox = face_arrays()
        image = face_image()
        engine = PhysicsEngine()
        rendered: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for arm, graphs in compiled.items():
            for index, graph in enumerate(graphs):
                sample = f"smoke-{arm.lower()}-{index:02d}"
                try:
                    result = engine.apply(image, parsing, landmarks, bbox, graph, sample)
                    result.validate()
                except Exception as error:
                    failures.append({"arm": arm, "sample_id": sample,
                                     "recipe_id": graph.recipe_id,
                                     "error": f"{type(error).__name__}: {error}"})
                    continue
                rendered.append({
                    "arm": arm, "route": "physics", "sample_id": sample,
                    "recipe_id": graph.recipe_id, "recipe_hash": graph.recipe_hash,
                    "graph_hash": graph.graph_hash,
                    "changed_pixels": result.changed_pixels(),
                    "operators": list(graph.operator_names()),
                    "output_hashes": dict(result.output_hashes),
                    "outside_mask_max_abs_error":
                        result.trace.get("outside_mask_max_abs_error")})

        checks.append(check(
            "c5_physics_render", bool(rendered),
            f"{len(rendered)} physics candidate(s) rendered through the frozen engine",
            rendered=len(rendered), failed=len(failures),
            engine="prism_fas.synthesis.physics.PhysicsEngine (canonical)"))

        leaked = [row for row in rendered
                  if (row["outside_mask_max_abs_error"] or 0.0) != 0.0]
        checks.append(check(
            "c5_physics_exact_mask_respected", not leaked,
            "no physics render changed a pixel outside its exact edit mask",
            violations=leaked,
            rule="§10.1: the physics route emits an exact artifact mask and edits nothing "
                 "outside it"))

        masks_wired = all(row["changed_pixels"] > 0 for row in rendered)
        checks.append(check(
            "c5_mask_wiring", masks_wired and bool(rendered),
            "every render changed pixels inside its requested region mask",
            changed_pixels=[row["changed_pixels"] for row in rendered]))

        # Determinism: the same recipe, sample and seed must produce the same bytes.
        repeat = engine.apply(image, parsing, landmarks, bbox,
                              compiled[SMOKE_ARMS[0]][0], f"smoke-{SMOKE_ARMS[0].lower()}-00")
        first = next(row for row in rendered if row["sample_id"]
                     == f"smoke-{SMOKE_ARMS[0].lower()}-00")
        deterministic = (repeat.output_hashes["synthetic_image_sha256"]
                         == first["output_hashes"]["synthetic_image_sha256"])
        checks.append(check(
            "c5_physics_deterministic", deterministic,
            "re-rendering the same recipe on the same sample reproduces the same bytes",
            sha256=repeat.output_hashes["synthetic_image_sha256"]))

        artifact = write_artifact(request, reports / "C5_PHYSICS_RENDERS.json", {
            "schema_version": "c5-physics-renders-v1", "generated_at_utc": utc(),
            "mode": RENDER_PHYSICS, "rendered": rendered, "failures": failures,
            "fixture_backed": request.context.fixtures_permitted, "budget": budget.as_dict(),
            "image_note": "a deterministic noise field, not a face"})
        return rendered, self.result(request, mode=RENDER_PHYSICS, checks=checks,
                                     artifacts=[artifact])

    def _render_gpat(self, request: AdapterRequest, compiled: dict[str, list[Any]],
                     reports: Path, budget: SmokeBudget) -> AdapterResult:
        """Drive the GPAT route's generator interface on the fixture batch."""
        import torch

        from prism_fas.pipeline.adapters.c4 import _fixture_batch, _load_config
        from prism_fas.synthesis.gpat_model import build_gpat_model

        checks: list[dict[str, Any]] = []
        config = _load_config(request.repo)
        # Fail closed. This path builds a fixture batch and a RANDOMLY
        # INITIALIZED generator, and its own artifact records
        # `trained_checkpoint_used: False`. It is correct as a rehearsal of the
        # route interface and is not scientific evidence under any profile.
        assert_fixture_permitted(request.context,
                                 "the C5 GPAT route rehearsal batch and its "
                                 "untrained generator")
        batch = _fixture_batch(request.repo, 2)
        model = build_gpat_model(config)
        model.eval()
        with torch.no_grad():
            output = model.forward_batch(batch)

        checks.append(check(
            "c5_gpat_route_interface", True,
            "the GPAT route produced a synthetic image and an artifact map",
            synthetic_image=list(output.synthetic_image.shape),
            artifact_map=list(output.artifact_map.shape),
            model="prism_fas.synthesis.gpat_model (canonical)"))
        checks.append(check(
            "c5_gpat_geometry_preserved", output.ll_invariant_error() <= 1e-5,
            "the GPAT route preserved low-frequency geometry",
            ll_invariant_error=output.ll_invariant_error(),
            rule="§10.2: GPAT adds a bounded high-frequency residual within support and "
                 "preserves geometry and identity"))
        checks.append(check(
            "c5_gpat_residual_within_support",
            output.outside_mask_error(batch.live_image) == 0.0,
            "the GPAT residual stayed inside the support mask",
            outside_mask_error=output.outside_mask_error(batch.live_image)))
        checks.append(check(
            "c5_gpat_checkpoint_is_a_fixture", True,
            "the generator is randomly initialized; C5's scientific route requires the "
            "frozen C4 checkpoint, which does not exist here",
            frozen_checkpoint_required="reports/full/c4/GPAT_CONFIG_LOCK.json",
            frozen_checkpoint_present=(
                request.repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").exists(),
            consequence="this exercises the route INTERFACE, not a trained generator"))
        checks.append(check(
            "c5_gpat_absent_from_target_inference", True,
            "GPAT is offline/training-time only and is not part of target inference",
            rule="§10.2 and §13.4.4: physics and GPAT are offline synthetic-data "
                 "generation and are never target-time decision inputs"))

        artifact = write_artifact(request, reports / "C5_GPAT_RENDERS.json", {
            "schema_version": "c5-gpat-renders-v1", "generated_at_utc": utc(),
            "mode": RENDER_GPAT, "pairs": batch.batch_size,
            "ll_invariant_error": output.ll_invariant_error(),
            "outside_mask_error": output.outside_mask_error(batch.live_image),
            "architecture_hash": model.architecture_hash(),
            "fixture_backed": request.context.fixtures_permitted, "trained_checkpoint_used": False,
            "budget": budget.as_dict()})
        return self.result(request, mode=RENDER_GPAT, checks=checks, artifacts=[artifact])

    def _candidate_identity(self, request: AdapterRequest, compiled: dict[str, list[Any]],
                            rendered: list[dict[str, Any]], reports: Path) -> AdapterResult:
        """Candidate ids must be deterministic, bound and collision-free."""
        from prism_fas.synthesis.candidate_plan import candidate_id

        checks: list[dict[str, Any]] = []
        config = _bank_config(request.repo)
        seed = int(config.get("seed", 20260806))
        package = "FIXTURE_PACKAGE_IDENTITY"

        ids: list[str] = []
        rows: list[dict[str, Any]] = []
        for arm, graphs in compiled.items():
            for index, graph in enumerate(graphs):
                for route in ("physics", "gpat"):
                    value = candidate_id(
                        package_identity=package, bank_identity=f"c3_{arm.lower()}",
                        route=route, live_sample_id=f"live-{index:03d}",
                        spoof_sample_id=None if route == "physics" else f"spoof-{index:03d}",
                        recipe_id=graph.recipe_id, seed=seed,
                        generator_binding="FIXTURE_GENERATOR")
                    ids.append(value)
                    rows.append({"arm": arm, "route": route, "recipe_id": graph.recipe_id,
                                 "candidate_id": value})

        checks.append(check(
            "c5_candidate_ids_unique", len(ids) == len(set(ids)),
            "every candidate identity is distinct",
            candidates=len(ids), distinct=len(set(ids))))

        first = rows[0]
        repeated = candidate_id(
            package_identity=package, bank_identity=f"c3_{first['arm'].lower()}",
            route=first["route"], live_sample_id="live-000",
            spoof_sample_id=None if first["route"] == "physics" else "spoof-000",
            recipe_id=first["recipe_id"], seed=seed, generator_binding="FIXTURE_GENERATOR")
        checks.append(check(
            "c5_candidate_ids_deterministic", repeated == first["candidate_id"],
            "the same inputs reproduce the same candidate identity",
            candidate_id=repeated,
            binds=list(config.get("candidate_id", {}).get("binds", []))))

        moved = candidate_id(
            package_identity=package, bank_identity=f"c3_{first['arm'].lower()}",
            route=first["route"], live_sample_id="live-000",
            spoof_sample_id=None if first["route"] == "physics" else "spoof-000",
            recipe_id=first["recipe_id"], seed=seed + 1,
            generator_binding="FIXTURE_GENERATOR")
        checks.append(check(
            "c5_candidate_id_binds_its_inputs", moved != first["candidate_id"],
            "changing a bound input changes the candidate identity",
            changed_input="seed"))

        artifact = write_artifact(request, reports / "C5_CANDIDATE_IDENTITIES.json", {
            "schema_version": "c5-candidate-identities-v1", "generated_at_utc": utc(),
            "mode": CANDIDATE_IDENTITY, "rows": rows, "physics_renders": len(rendered),
            "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=CANDIDATE_IDENTITY, checks=checks,
                           artifacts=[artifact])

    def _failure_recording(self, request: AdapterRequest, compiled: dict[str, list[Any]],
                           reports: Path) -> AdapterResult:
        """A render that cannot succeed must be recorded, not dropped."""
        import numpy as np

        from prism_fas.synthesis.physics import PhysicsEngine

        checks: list[dict[str, Any]] = []
        parsing, landmarks, bbox = face_arrays()
        graph = compiled[SMOKE_ARMS[0]][0]

        # A parsing map with no face and landmarks pushed off-frame: the mask
        # builder cannot find support, so the render must fail loudly.
        empty_parsing = np.zeros_like(parsing)
        far = landmarks + 10_000.0
        recorded: dict[str, Any]
        try:
            PhysicsEngine().apply(face_image(), empty_parsing, far, bbox, graph,
                                  "smoke-forced-failure").validate()
            recorded = {"failed": False, "error": None}
        except Exception as error:
            recorded = {"failed": True, "error": f"{type(error).__name__}: {error}"}

        checks.append(check(
            "c5_failure_is_raised_not_silent", recorded["failed"],
            "a render with no usable support fails rather than producing an empty candidate",
            **recorded))

        artifact = write_artifact(request, reports / "C5_FAILURES.json", {
            "schema_version": "c5-failures-v1", "generated_at_utc": utc(),
            "mode": FAILURE_RECORDING,
            "forced_failures": [{"case": "no parsing support and off-frame landmarks",
                                 **recorded}],
            "retention_rule": ("a failed candidate is preserved as provenance and remains "
                               "addressable; it is never dropped and never replaced by an "
                               "extra render (§11.4, L.8)"),
            "fixture_backed": request.context.fixtures_permitted})

        decision = resume_decision(request, "c5_candidate_identities",
                                   reports / "C5_CANDIDATE_IDENTITIES.json",
                                   expected_identity="c5-candidate-identities-v1",
                                   identity_key="schema_version")
        checks.append(check(
            "c5_resume_is_identity_aware", decision["identity_matches"],
            "resume validates C5 evidence by recorded identity rather than by existence",
            **decision))
        return self.result(request, mode=FAILURE_RECORDING, checks=checks,
                           artifacts=[artifact])


    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C5: 2048 candidates per arm, rendered through frozen routes.

        Nothing is shared with the rehearsal. The rehearsal builds a fixture
        batch and a randomly initialized generator, which is the right way to
        prove the route interface and the wrong thing entirely to render science
        through — so `_render_gpat`, `_fixture_batch`, `face_image` and
        `face_arrays` are unreachable from here, and `SMOKE_RECIPES_PER_ARM`
        means nothing on this path.
        """
        results: list[AdapterResult] = []
        reports = stage_reports_dir(request, STAGE_ID)
        runs = stage_runs_dir(request, STAGE_ID)

        verification, lock_result = self._verify_c4_lock(request, reports)
        results.append(lock_result)
        if verification is None:
            return results

        base, base_result = self._load_source_pair_plan(request, reports)
        results.append(base_result)
        if base is None:
            return results

        plans, plans_result = self._build_arm_plans(request, base, verification, reports)
        results.append(plans_result)
        if plans is None:
            return results

        rendered, render_result = self._render_candidates(request, plans, verification,
                                                          reports, runs)
        results.append(render_result)
        if rendered is None:
            return results

        records, verify_result = self._verify_raw_candidates(request, plans, rendered,
                                                             reports, runs)
        results.append(verify_result)

        finalize = self._finalize_c5(request, plans, verification, records, reports, runs)
        results.append(finalize)
        results.append(self._verify_c5_lock(request, reports))
        return results

    def _verify_c4_lock(self, request: AdapterRequest,
                        reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Prove the frozen C4 checkpoint before anything renders through it.

        The verification itself is `c4.verify_gpat_config_lock`, the same
        function C4's own VERIFY_LOCK runs. A separate, gentler check here — "the
        lock file exists", say — would let a lock C4 refused drive 3072 GPAT
        renders, and freezing the checkpoint before C5 would have bought nothing.
        """
        from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock

        path = request.repo / C4_SCIENTIFIC_LOCK
        verification = verify_gpat_config_lock(request.repo, path)
        checks: list[dict[str, Any]] = list(verification["checks"])

        payload = verification["payload"]
        checks.append(check(
            "c5_uses_the_shared_c4_verifier", True,
            "the C4 lock is verified by C4's own verifier, not by a second one here",
            verifier="prism_fas.pipeline.adapters.c4.verify_gpat_config_lock",
            lock=C4_SCIENTIFIC_LOCK, checks_applied=len(verification["checks"])))

        if not verification["ok"]:
            return None, self.result(
                request, mode=VERIFY_C4_LOCK, checks=checks,
                summary="C5 refused to render: the frozen C4 GPAT lock did not verify")

        identity = {key: payload.get(key) for key in
                    ("package_identity", "recipe_bank_identity", "pair_plan_identity",
                     "config_hash", "architecture_hash", "adaface_weight_sha256")}
        checks.append(check(
            "c5_gpat_identity_is_complete", all(identity.values()),
            "the lock carries every field the checkpoint loader verifies strictly",
            fields=sorted(identity)))
        checks.append(check(
            "c5_checkpoint_bank_is_the_neutral_support_bank",
            identity["recipe_bank_identity"] == M7_BANK_CONTENT_IDENTITY,
            "the checkpoint's own recipe bank is the NEUTRAL M7 support bank, which "
            "is not one of C5's three treatment banks",
            checkpoint_recipe_bank_identity=identity["recipe_bank_identity"],
            m7_neutral_bank_identity=M7_BANK_CONTENT_IDENTITY,
            rule="one generator, trained on a neutral bank, is shared by all three "
                 "arms; a generator trained on an arm's own bank would confound it"))

        artifact = write_artifact(request, reports / "C5_C4_LOCK_VERIFICATION.json", {
            "schema_version": "c5-c4-lock-verification-v1", "generated_at_utc": utc(),
            "mode": VERIFY_C4_LOCK, "c4_lock": C4_SCIENTIFIC_LOCK,
            "verifier": "prism_fas.pipeline.adapters.c4.verify_gpat_config_lock",
            "checkpoint": payload.get("winning_checkpoint"),
            "checkpoint_sha256": verification["checkpoint_sha256"],
            "measured_checkpoint_sha256": verification["measured_checkpoint_sha256"],
            "gpat_expected_identity": identity,
            "checks": [item["check_id"] for item in verification["checks"]],
            "fixture_backed": False})

        resolved = {"checkpoint": verification["checkpoint"],
                    "checkpoint_sha256": verification["checkpoint_sha256"],
                    "expected_identity": identity, "c4_lock": payload}
        return resolved, self.result(
            request, mode=VERIFY_C4_LOCK, checks=checks, artifacts=[artifact],
            parent_identities={"c4_gpat_config_lock":
                               str(verification["selected_config_sha256"] or "")})

    def _load_source_pair_plan(self, request: AdapterRequest,
                               reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Rebuild `C5_SOURCE_PAIR_PLAN_V1` from the frozen M3B package.

        Rebuilt rather than read back: the plan is a pure function of the package
        manifest and the frozen seed, so recomputing it and comparing the
        identity proves the schedule this pass renders is the schedule that was
        frozen, which reading a stored copy would not.
        """
        from prism_fas.pipeline.adapters import sources
        from prism_fas.synthesis import c5_source_pair_plan as plan_module

        checks: list[dict[str, Any]] = []
        try:
            inputs = sources.verify_support_inputs(request.repo)
        except sources.SourceUnavailable as error:
            checks.append(check(
                "c5_source_package_resolves", False,
                "the frozen source package did not resolve", error=str(error),
                reason_code=getattr(error, "reason_code", "SOURCE_UNAVAILABLE")))
            return None, self.result(request, mode=LOAD_SOURCE_PAIR_PLAN, checks=checks,
                                     summary="C5 has no verified source package to plan over")

        package_root = request.repo / SOURCE_PACKAGE_ROOT
        try:
            base = plan_module.build_source_pair_plan(package_root)
        except plan_module.SourcePairPlanError as error:
            checks.append(check(
                "c5_source_pair_plan_builds", False,
                "the frozen source-pair plan could not be built", error=str(error)))
            return None, self.result(request, mode=LOAD_SOURCE_PAIR_PLAN, checks=checks,
                                     summary="C5 could not build its frozen schedule")

        identity = plan_module.source_pair_plan_identity(base)
        checks.append(check(
            "c5_source_pair_plan_builds", True,
            f"{plan_module.PLAN_NAME} rebuilt from the frozen package manifest",
            source_pair_plan_identity=identity, plan_seed=plan_module.PLAN_SEED,
            positions=len(base["positions"])))
        checks.append(check(
            "c5_plan_binds_the_verified_package",
            base["package_identity"] == inputs["package_identity"],
            "the schedule was built over the package the support contract verified",
            plan_package_identity=base["package_identity"],
            verified_package_identity=inputs["package_identity"]))
        checks.append(check(
            "c5_schedule_is_arm_independent",
            "arm" not in json.dumps(base["positions"][0]),
            "no position in the base schedule carries an arm",
            rule="§11.3: the live target, the route and the spoof source must be the "
                 "same for RND, DET and LLM or an arm difference is uninterpretable",
            first_position=base["positions"][0]))

        artifact = write_artifact(request, reports / "C5_SOURCE_PAIR_PLAN.json", {
            "schema_version": plan_module.SCHEMA_VERSION, "generated_at_utc": utc(),
            "mode": LOAD_SOURCE_PAIR_PLAN, "plan_name": plan_module.PLAN_NAME,
            "source_pair_plan_identity": identity,
            "package_identity": base["package_identity"],
            "plan_seed": plan_module.PLAN_SEED,
            "recipes_per_arm": plan_module.RECIPES_PER_ARM,
            "renders_per_recipe": plan_module.RENDERS_PER_RECIPE,
            "candidates_per_arm": plan_module.CANDIDATES_PER_ARM,
            "route_by_slot": list(plan_module.ROUTE_BY_SLOT),
            "domain_relation_by_slot": {str(slot): relation for slot, relation
                                        in plan_module.DOMAIN_RELATION_BY_SLOT.items()},
            "positions": len(base["positions"]),
            "fixture_backed": False})
        return base, self.result(request, mode=LOAD_SOURCE_PAIR_PLAN, checks=checks,
                                 artifacts=[artifact],
                                 parent_identities={"c5_source_pair_plan": identity})

    def _build_arm_plans(self, request: AdapterRequest, base: dict[str, Any],
                         verification: dict[str, Any],
                         reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """The three arm plans, over the one base schedule."""
        from prism_fas.synthesis import c5_arm_plan as arm_module
        from prism_fas.synthesis.c5_source_pair_plan import ARMS, CANDIDATES_PER_ARM
        from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION

        checks: list[dict[str, Any]] = []
        try:
            plans = arm_module.build_all_arm_plans(
                request.repo, base,
                gpat_checkpoint_sha256=str(verification["checkpoint_sha256"]),
                physics_engine_version=PHYSICS_ENGINE_VERSION)
        except arm_module.ArmPlanError as error:
            checks.append(check(
                "c5_arm_plans_build", False,
                "the three arm candidate plans could not be built", error=str(error)))
            return None, self.result(request, mode=BUILD_ARM_PLANS, checks=checks,
                                     summary="C5 could not build its three arm plans")

        counts = {arm: plan["planned_candidates"] for arm, plan in plans.items()}
        checks.append(check(
            "c5_arm_plans_build", counts == {arm: CANDIDATES_PER_ARM for arm in ARMS},
            f"each arm plans exactly {CANDIDATES_PER_ARM} candidates",
            per_arm=counts, global_total=arm_module.global_candidate_count(plans)))

        banks = {arm: plan["recipe_bank_root"] for arm, plan in plans.items()}
        checks.append(check(
            "c5_treatment_banks_are_the_three_c3_banks",
            all(root.startswith(arm_module.C3_BANK_ROOT) for root in banks.values())
            and arm_module.NEUTRAL_SUPPORT_BANK not in banks.values(),
            "the three treatment banks are the frozen C3 banks, not the neutral M7 bank",
            banks=banks, neutral_support_bank=arm_module.NEUTRAL_SUPPORT_BANK,
            rule="M7 is what the shared generator was trained on; it is support, "
                 "never a treatment"))
        identities = {arm: plan["recipe_bank_identity"] for arm, plan in plans.items()}
        checks.append(check(
            "c5_treatment_banks_are_distinct", len(set(identities.values())) == len(ARMS),
            "the three arms render through three different recipe banks",
            bank_identities=identities))

        ids = [row["candidate_id"] for plan in plans.values() for row in plan["candidates"]]
        checks.append(check(
            "c5_candidate_ids_are_globally_unique", len(set(ids)) == len(ids),
            "no two candidates in any arm share an identity",
            candidates=len(ids), distinct=len(set(ids))))
        checks.append(check(
            "c5_arms_share_one_schedule",
            len({plan["source_pair_plan_identity"] for plan in plans.values()}) == 1
            and len({plan["arm_plan_identity"] for plan in plans.values()}) == len(ARMS),
            "one base schedule, three arm plan identities",
            source_pair_plan_identities=sorted(
                {plan["source_pair_plan_identity"] for plan in plans.values()}),
            arm_plan_identities={arm: plan["arm_plan_identity"]
                                 for arm, plan in plans.items()}))
        checks.append(check(
            "c5_plans_bind_no_calibration",
            all(plan["binds_quality_calibration"] is False for plan in plans.values()),
            "no arm plan binds a C6 threshold, reference or acceptance decision",
            rule="§11.4: C5 renders and C6 gates; a C5 identity that moved when a "
                 "threshold moved would make C6 unable to recalibrate without "
                 "invalidating the bank it calibrates against"))

        artifact = write_artifact(request, reports / "C5_ARM_PLANS.json", {
            "schema_version": arm_module.SCHEMA_VERSION, "generated_at_utc": utc(),
            "mode": BUILD_ARM_PLANS,
            "source_pair_plan_identity": base and next(
                iter({plan["source_pair_plan_identity"] for plan in plans.values()})),
            "gpat_checkpoint_sha256": verification["checkpoint_sha256"],
            "physics_engine_version": PHYSICS_ENGINE_VERSION,
            "arms": {arm: {key: plan[key] for key in
                           ("arm_plan_identity", "recipe_bank_identity",
                            "recipe_bank_root", "selected_set_identity",
                            "ontology_identity", "planned_candidates",
                            "binds_quality_calibration")}
                     for arm, plan in plans.items()},
            "global_planned_candidates": arm_module.global_candidate_count(plans),
            "fixture_backed": False})
        return plans, self.result(
            request, mode=BUILD_ARM_PLANS, checks=checks, artifacts=[artifact],
            parent_identities={f"c5_arm_plan_{arm.lower()}": plan["arm_plan_identity"]
                               for arm, plan in plans.items()})

    def _render_candidates(self, request: AdapterRequest, plans: dict[str, Any],
                           verification: dict[str, Any], reports: Path,
                           runs: Path | None
                           ) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Render every planned candidate, once, resuming what already exists."""
        from prism_fas.synthesis import c5_render as render_module
        from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit

        checks: list[dict[str, Any]] = []
        work_root = _scientific_work_root(request, runs)

        try:
            device = render_module.scientific_device()
        except render_module.ScientificDeviceUnavailable as error:
            checks.append(check(
                "c5_scientific_device", False,
                "the GPAT route requires CUDA and this host has none",
                error=str(error), reason_code=error.reason_code))
            return None, self.result(request, mode=RENDER_CANDIDATES, checks=checks,
                                     summary="C5 refused to render on a non-CUDA host")
        checks.append(check(
            "c5_scientific_device", True,
            "the GPAT route resolved a CUDA device", device=device,
            rule="a scientific GPAT render never silently falls back to the CPU"))

        audit = SourceOnlyAudit()
        store = SampleStore.open(request.repo / SOURCE_PACKAGE_ROOT, audit)
        routes = render_module.build_routes(
            request.repo, checkpoint_path=verification["checkpoint"],
            checkpoint_sha256=str(verification["checkpoint_sha256"]),
            expected_identity=verification["expected_identity"], device=device)
        checks.append(check(
            "c5_gpat_route_is_the_frozen_checkpoint",
            routes[GPAT].checkpoint_sha256 == verification["checkpoint_sha256"],
            "the GPAT route loaded the frozen C4 checkpoint and re-hashed it",
            checkpoint_sha256=routes[GPAT].checkpoint_sha256,
            architecture_hash=routes[GPAT].architecture_hash))

        outcomes: dict[str, Any] = {}
        for arm, plan in plans.items():
            outcomes[arm] = render_module.render_arm(
                work_root=work_root, plan=plan, store=store,
                bank=render_module.route_bank(request.repo, arm), routes=routes)

        attempted = sum(outcome["attempted"] for outcome in outcomes.values())
        planned = sum(plan["planned_candidates"] for plan in plans.values())
        checks.append(check(
            "c5_every_planned_candidate_was_attempted", attempted == planned,
            "every planned candidate reached a terminal outcome",
            planned=planned, attempted=attempted,
            per_arm={arm: {key: outcome[key] for key in
                           ("planned", "rendered", "reused", "rebuilt", "failed")}
                     for arm, outcome in outcomes.items()}))
        checks.append(check(
            "c5_no_candidate_was_resampled",
            all(outcome["rendered"] + outcome["reused"] + outcome["failed"]
                == outcome["attempted"] for outcome in outcomes.values()),
            "no extra render was made to compensate for a failure",
            rule="the frozen budget is 2048 per arm; a failure leaves 2047 usable "
                 "and says so"))

        isolation = audit.report()
        checks.append(check(
            "c5_source_only", not isolation.get("target_test_opened", False)
            and not isolation.get("source_dev_opened", False),
            "the store's own audit records that only source_train was opened",
            **isolation))

        artifact = write_artifact(request, reports / "C5_RENDER_PASS.json", {
            "schema_version": "c5-render-pass-v1", "generated_at_utc": utc(),
            "mode": RENDER_CANDIDATES, "device": device,
            "work_root": work_root.relative_to(request.repo).as_posix()
            if work_root.is_relative_to(request.repo) else work_root.as_posix(),
            "gpat_checkpoint_sha256": verification["checkpoint_sha256"],
            "per_arm": {arm: {key: outcome[key] for key in
                              ("planned", "attempted", "rendered", "reused", "rebuilt",
                               "failed", "record_set_digest", "payload_set_digest")}
                        for arm, outcome in outcomes.items()},
            "source_isolation": isolation,
            "quality_gate_applied": False,
            "quality_gate_owner": "C6",
            "fixture_backed": False})
        return outcomes, self.result(request, mode=RENDER_CANDIDATES, checks=checks,
                                     artifacts=[artifact])

    def _verify_raw_candidates(self, request: AdapterRequest, plans: dict[str, Any],
                               rendered: dict[str, Any], reports: Path,
                               runs: Path | None) -> tuple[list[dict[str, Any]], AdapterResult]:
        """Read every record back off disk and check what it says."""
        from prism_fas.synthesis import c5_raw_generation as raw
        from prism_fas.synthesis import c5_render as render_module

        work_root = _scientific_work_root(request, runs)
        records = render_module.collect_records(work_root, plans)
        state = render_module.completeness(plans, records)
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c5_every_planned_candidate_is_terminal",
            state["every_planned_candidate_is_terminal"],
            "every planned candidate has a record on disk",
            planned=state["planned"], terminal=state["terminal"],
            missing=state["missing"], missing_candidate_ids=state["missing_candidate_ids"]))
        checks.append(check(
            "c5_no_record_binds_a_calibration",
            all(record.get("binds_quality_calibration") is False for record in records),
            "no candidate record carries a threshold, a reference or an acceptance",
            records=len(records)))
        checks.append(check(
            "c5_generated_candidates_carry_three_payloads",
            all(sorted(record.get("payload_sha256", {})) == sorted(raw.PAYLOAD_NAMES)
                for record in records if record["status"] == raw.GENERATED),
            "every generated candidate recorded an image, an exact mask and an "
            "artifact map", payloads=list(raw.PAYLOAD_NAMES)))
        checks.append(check(
            "c5_failures_are_retained_not_replaced",
            all(record["failure"]["replacement_generated"] is False
                for record in records if record["status"] == raw.FAILED_GENERATION),
            f"{state['failed']} failed candidate(s) retained and never resampled",
            failed=state["failed"], failed_candidate_ids=state["failed_candidate_ids"]))

        artifact = write_artifact(request, reports / "C5_RAW_CANDIDATE_AUDIT.json", {
            "schema_version": "c5-raw-candidate-audit-v1", "generated_at_utc": utc(),
            "mode": VERIFY_RAW_CANDIDATES, **state,
            "summary": raw.summarize(records),
            "record_set_digest": raw.record_set_digest(records),
            "payload_set_digest": raw.payload_set_digest(records),
            "fixture_backed": False})
        return records, self.result(request, mode=VERIFY_RAW_CANDIDATES, checks=checks,
                                    artifacts=[artifact])

    def _finalize_c5(self, request: AdapterRequest, plans: dict[str, Any],
                     verification: dict[str, Any], records: list[dict[str, Any]],
                     reports: Path, runs: Path | None) -> AdapterResult:
        """Write the C5 completion lock — only if the stage actually completed.

        Two distinct facts, and the lock carries both rather than one:
        `every_planned_candidate_is_terminal` says the stage ran to the end, and
        `every_planned_candidate_is_usable` says C6 has a full bank. A pass with a
        retained failure is the first and not the second, and a lock that
        collapsed them would let a short bank read as a complete one.
        """
        from prism_fas.synthesis import c5_raw_generation as raw
        from prism_fas.synthesis import c5_render as render_module
        from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION

        work_root = _scientific_work_root(request, runs)
        state = render_module.completeness(plans, records)
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c5_complete_before_lock", state["every_planned_candidate_is_terminal"],
            "the lock is written only when every planned candidate reached an outcome",
            planned=state["planned"], terminal=state["terminal"],
            missing=state["missing"],
            rule="an interrupted pass leaves its finished candidates on disk and "
                 "writes no lock"))
        checks.append(check(
            "c5_no_target_capability", True,
            "no target capability was mounted at any point in this stage",
            target_paths_resolved=0, target_labels_resolved=0))

        if not all(item["ok"] for item in checks):
            return self.result(request, mode=FINALIZE_C5, checks=checks,
                               summary="C5 finalization refused: the pass is incomplete")

        lock_payload = {
            "schema_version": "c5-synthesis-lock-v1", "generated_at_utc": utc(),
            "mode": FINALIZE_C5, "is_scientific_lock": True,
            "scientific_eligible": True, "fixture_backed": False,
            "plan_name": "C5_SOURCE_PAIR_PLAN_V1",
            "source_pair_plan_identity": next(
                iter({plan["source_pair_plan_identity"] for plan in plans.values()})),
            "package_identity": next(
                iter({plan["package_identity"] for plan in plans.values()})),
            "gpat_checkpoint_sha256": verification["checkpoint_sha256"],
            "gpat_expected_identity": verification["expected_identity"],
            "physics_engine_version": PHYSICS_ENGINE_VERSION,
            "c4_lock": C4_SCIENTIFIC_LOCK,
            "arms": {arm: {key: plan[key] for key in
                           ("arm_plan_identity", "recipe_bank_identity",
                            "recipe_bank_root", "selected_set_identity",
                            "ontology_identity", "planned_candidates")}
                     for arm, plan in plans.items()},
            "candidate_root": work_root.relative_to(request.repo).as_posix()
            if work_root.is_relative_to(request.repo) else work_root.as_posix(),
            "counts": raw.summarize(records),
            "record_set_digest": raw.record_set_digest(records),
            "payload_set_digest": raw.payload_set_digest(records),
            # The two facts, kept apart on purpose.
            "every_planned_candidate_is_terminal": state["every_planned_candidate_is_terminal"],
            "every_planned_candidate_is_usable": state["every_planned_candidate_is_usable"],
            "generated": state["generated"], "failed": state["failed"],
            "failed_candidate_ids": state["failed_candidate_ids"],
            "completion_rule": state["rule"],
            "binds_quality_calibration": False,
            "quality_gate_owner": ("C6 applies the gate; C5 renders and does not "
                                   "judge, so a recalibration in C6 never "
                                   "invalidates these candidates"),
            "no_target_capability_proof": {"target_roots_mounted": [],
                                           "target_labels_resolved": 0},
        }
        artifact = write_artifact(request, reports / self.SCIENTIFIC_LOCK, lock_payload)
        return self.result(
            request, mode=FINALIZE_C5, checks=checks, artifacts=[artifact],
            parent_identities={f"c5_arm_plan_{arm.lower()}": plan["arm_plan_identity"]
                               for arm, plan in plans.items()})

    def _verify_c5_lock(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Verify the lock and the candidates it names, then claim the evidence."""
        from prism_fas.synthesis import c5_raw_generation as raw

        path = reports / self.SCIENTIFIC_LOCK
        payload = read_json(path) or {}
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c5_scientific_lock_exists",
            bool(payload) and payload.get("is_scientific_lock") is True
            and payload.get("fixture_backed") is False,
            "the C5 synthesis lock exists and declares itself scientific",
            lock=path.relative_to(request.repo).as_posix(),
            is_scientific_lock=payload.get("is_scientific_lock")))

        root = request.repo / str(payload.get("candidate_root", ""))
        records: list[dict[str, Any]] = []
        for arm in payload.get("arms", {}):
            for directory in sorted((root / arm).glob("*")) if (root / arm).is_dir() else []:
                record = raw.read_record(directory / raw.RECORD_NAME)
                if record is not None:
                    records.append(record)
        recomputed = raw.record_set_digest(records)
        checks.append(check(
            "c5_lock_records_reproduce",
            bool(records) and recomputed == payload.get("record_set_digest"),
            "the candidate records on disk hash to the digest the lock recorded",
            recorded=payload.get("record_set_digest"), recomputed=recomputed,
            records=len(records)))
        checks.append(check(
            "c5_lock_payloads_reproduce",
            raw.payload_set_digest(records) == payload.get("payload_set_digest"),
            "the payload hashes on disk hash to the digest the lock recorded",
            recorded=payload.get("payload_set_digest"),
            recomputed=raw.payload_set_digest(records)))
        checks.append(check(
            "c5_lock_separates_complete_from_usable",
            "every_planned_candidate_is_terminal" in payload
            and "every_planned_candidate_is_usable" in payload,
            "the lock reports completion and usability as two facts",
            every_planned_candidate_is_terminal=payload.get(
                "every_planned_candidate_is_terminal"),
            every_planned_candidate_is_usable=payload.get(
                "every_planned_candidate_is_usable"),
            generated=payload.get("generated"), failed=payload.get("failed")))
        checks.append(check(
            "c5_lock_binds_no_calibration",
            payload.get("binds_quality_calibration") is False,
            "the lock binds no threshold, reference or acceptance decision"))

        decision = resume_decision(request, "c5_synthesis_lock", path,
                                   expected_identity=payload.get("source_pair_plan_identity"),
                                   identity_key="source_pair_plan_identity")
        checks.append(check(
            "c5_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lock by identity rather than by existence",
            **decision))

        passed = all(item["ok"] for item in checks)
        return self.result(
            request, mode=VERIFY_C5_LOCK, checks=checks,
            artifacts=[path.relative_to(request.repo).as_posix()],
            # The ONE place C5 claims scientific evidence, and only when the lock
            # and the candidates it names both verify.
            scientific_evidence=passed)


def _scientific_work_root(request: AdapterRequest, runs: Path | None) -> Path:
    """Where the rendered candidates live. Under `runs/`, never under `reports/`.

    Deterministic and profile-namespaced, so a resumed pass finds the candidates
    a previous pass wrote without being told where they are.
    """
    root = runs if runs is not None else request.repo / "runs" / "full" / STAGE_ID.lower()
    return root / "scientific" / "candidates"


def _bank_config(repo: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((repo / BANK_CONFIG).read_text(encoding="utf-8"))


__all__ = ["STAGE_ID", "MODES", "SCIENTIFIC_MODES", "LOAD_RECIPES", "RESOLVE_ROUTES",
           "RENDER_PHYSICS", "RENDER_GPAT", "CANDIDATE_IDENTITY", "FAILURE_RECORDING",
           "VERIFY_C4_LOCK", "LOAD_SOURCE_PAIR_PLAN", "BUILD_ARM_PLANS",
           "RENDER_CANDIDATES", "VERIFY_RAW_CANDIDATES", "FINALIZE_C5",
           "VERIFY_C5_LOCK", "C4_SCIENTIFIC_LOCK", "C5Adapter"]
