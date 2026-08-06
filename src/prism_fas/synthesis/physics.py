from __future__ import annotations
from typing import Any
import numpy as np
from prism_fas.recipes.compile import CompiledRecipeGraph, local_rng
from .contracts import PhysicsResult, SynthesisError, array_hash, mask_hash, validate_image
from .masks import RegionMaskBuilder
from .operators import build_operator

PHYSICS_ENGINE_VERSION = "m7-physics-v1"


class PhysicsEngine:
    """CPU physics engine with exact masks.

    Applies a compiled operator graph to one real live crop and returns the
    synthetic image together with the exact edit mask. The final composite is
    performed against the original image, so the outside-mask difference is
    exactly zero rather than merely small.
    """
    version = PHYSICS_ENGINE_VERSION

    def __init__(self, *, coverage_tolerance: float = 0.05):
        self.coverage_tolerance = float(coverage_tolerance)

    def apply(self, image: np.ndarray, parsing: np.ndarray | None, landmarks: np.ndarray, bbox: np.ndarray,
              compiled_recipe: CompiledRecipeGraph, sample_id: str, *, crop_box: np.ndarray | None = None) -> PhysicsResult:
        original = validate_image(image, name="input image")
        height, width = original.shape[1], original.shape[2]
        graph = compiled_recipe
        if not graph.nodes: raise SynthesisError(f"{graph.recipe_id}: compiled graph has no operator nodes")

        # 1. deterministic requested region mask
        builder = RegionMaskBuilder(height=height, width=width,
                                    parsing=None if parsing is None else np.asarray(parsing),
                                    landmarks=np.asarray(landmarks, dtype=np.float32),
                                    bbox=np.asarray(bbox, dtype=np.float32),
                                    crop_box=None if crop_box is None else np.asarray(crop_box, dtype=np.float32))
        policy = graph.region_mask_policy
        mask_seed = graph.node_seed(graph.nodes[0], f"{sample_id}|region_mask")
        masks = builder.build(list(graph.requested_regions), geometry_shape=str(policy.get("geometry_shape", "flat")),
                              coverage=float(policy.get("requested_coverage", 1.0)), seed=mask_seed,
                              coverage_tolerance=float(policy.get("coverage_tolerance", self.coverage_tolerance)))
        requested = np.asarray(masks.requested_region_mask, dtype=np.float32)
        support = np.asarray(masks.operator_support_mask, dtype=np.float32)[0].astype(bool)

        # 2-3. apply operators in graph order, tracking the union of actual supports
        working = original
        union = np.zeros((height, width), dtype=bool)
        per_operator: dict[str, np.ndarray] = {}
        operator_trace: list[dict[str, Any]] = []
        for node in graph.nodes:
            operator = build_operator(node.operator_name)
            seed = graph.node_seed(node, sample_id)
            result = operator.apply(working, support.astype(np.float32)[None], node.strength, graph.capture,
                                    local_rng(seed), dict(node.parameters), seed=seed)
            actual = np.asarray(result.actual_support_mask, dtype=np.float32)[0].astype(bool)
            union |= actual
            per_operator[node.operator_name] = actual.astype(np.float32)[None]
            working = result.image
            operator_trace.append({"node_index": node.node_index, "operator": node.operator_name,
                                   "operator_seed": int(seed), "strength": float(node.strength),
                                   "support_policy": node.support_policy,
                                   "support_pixels": int(actual.sum()),
                                   "parameters_used": result.parameters_used,
                                   "input_key": node.input_key, "output_key": node.output_key,
                                   "trace": result.trace})

        # 4-5. exact edit mask and exact support composite against the ORIGINAL
        exact = union
        if not exact.any(): raise SynthesisError(f"{graph.recipe_id}: no operator declared any support")
        output = np.where(exact[None, :, :], working, original).astype(np.float32)
        output = np.ascontiguousarray(np.clip(output, np.float32(0.0), np.float32(1.0)))

        # 6. artifact strength map from the actual per-pixel RGB difference
        difference = np.abs(output - original).mean(axis=0).astype(np.float32)
        peak = float(difference.max())
        strength_map = (difference / np.float32(peak)) if peak > 0.0 else np.zeros_like(difference)
        strength_map = np.clip(strength_map, np.float32(0.0), np.float32(1.0)).astype(np.float32)
        strength_map = np.where(exact, strength_map, np.float32(0.0)).astype(np.float32)[None]

        # 7. verify exact outside-mask preservation
        outside = ~exact
        outside_error = float(np.abs(output - original)[:, outside].max()) if outside.any() else 0.0
        if outside_error != 0.0:
            raise SynthesisError(f"{graph.recipe_id}/{sample_id}: outside-mask difference {outside_error} is not exactly zero")
        changed = int((np.abs(output - original).max(axis=0) > 0).sum())

        result = PhysicsResult(
            synthetic_image=output, exact_edit_mask=exact.astype(np.float32)[None], artifact_strength_map=strength_map,
            requested_region_mask=requested, per_operator_support_masks=per_operator, recipe_id=graph.recipe_id,
            recipe_hash=graph.recipe_hash, graph_hash=graph.graph_hash, sample_id=sample_id,
            trace={"engine_version": self.version, "compiler_version": graph.compiler_version,
                   "bank_id": graph.bank_id, "recipe_id": graph.recipe_id, "recipe_hash": graph.recipe_hash,
                   "graph_hash": graph.graph_hash, "sample_id": sample_id,
                   "ontology_version": graph.ontology_version, "ontology_sha256": graph.ontology_sha256,
                   "requested_regions": list(graph.requested_regions), "region_sources": masks.region_sources,
                   "requested_coverage": masks.requested_coverage, "achieved_coverage": masks.achieved_coverage,
                   "mask_metadata": masks.metadata, "region_mask_seed": int(mask_seed),
                   "operators": operator_trace, "operator_order": list(graph.operator_names()),
                   "exact_edit_pixels": int(exact.sum()), "changed_pixels": changed,
                   "outside_mask_max_abs_error": outside_error,
                   "max_abs_difference_inside": round(float(peak), 8)},
            output_hashes={"input_image_sha256": array_hash(original), "synthetic_image_sha256": array_hash(output),
                           "exact_edit_mask_sha256": mask_hash(exact),
                           "requested_region_mask_sha256": mask_hash(np.asarray(requested)[0].astype(bool)),
                           "artifact_strength_map_sha256": array_hash(strength_map)})
        return result.validate()
