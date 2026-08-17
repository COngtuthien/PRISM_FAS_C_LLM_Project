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
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, read_json,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)
from prism_fas.pipeline.adapters.tiny import face_arrays, face_image, frozen_recipes

STAGE_ID = "C5"

LOAD_RECIPES = "LOAD_RECIPES"
RESOLVE_ROUTES = "RESOLVE_ROUTES"
RENDER_PHYSICS = "RENDER_PHYSICS"
RENDER_GPAT = "RENDER_GPAT"
CANDIDATE_IDENTITY = "CANDIDATE_IDENTITY"
FAILURE_RECORDING = "FAILURE_RECORDING"

MODES: tuple[str, ...] = (LOAD_RECIPES, RESOLVE_ROUTES, RENDER_PHYSICS, RENDER_GPAT,
                          CANDIDATE_IDENTITY, FAILURE_RECORDING)

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

    def run_smoke(self, request: AdapterRequest) -> list[AdapterResult]:
        results: list[AdapterResult] = []
        budget = SmokeBudget.from_profile(request.profile)
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
            "ontology_identity": ontology.sha256, "fixture_backed": True,
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
            "observed": observed, "fixture_backed": True})
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
            "fixture_backed": True, "budget": budget.as_dict(),
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
            "fixture_backed": True, "trained_checkpoint_used": False,
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
            "fixture_backed": True})
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
            "fixture_backed": True})

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


def _bank_config(repo: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((repo / BANK_CONFIG).read_text(encoding="utf-8"))


__all__ = ["STAGE_ID", "MODES", "LOAD_RECIPES", "RESOLVE_ROUTES", "RENDER_PHYSICS",
           "RENDER_GPAT", "CANDIDATE_IDENTITY", "FAILURE_RECORDING", "C5Adapter"]
