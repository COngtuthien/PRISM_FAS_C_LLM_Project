from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np
from .canonical import canonical_json, recipe_hash, stable_hash
from .conditioning import CONDITIONING_DIM, CONDITIONING_VERSION, conditioning_vector, feature_names, feature_names_sha256
from .ontology import Ontology
from .schema import RecipeV11

COMPILER_VERSION = "m7-compiler-v1"
GRAPH_SCHEMA_VERSION = "m7-graph-v1"
MASK_POLICY = "parsing_first_geometry_fallback"
DEFAULT_COVERAGE_TOLERANCE = 0.05
SEED_MODULUS = 2 ** 32


class CompileError(ValueError):
    """A recipe cannot be compiled into an operator graph."""


def operator_application_order() -> tuple[str, ...]:
    """Physically motivated, fixed operator order. Lazy import keeps the recipe
    package importable without the synthesis package."""
    from prism_fas.synthesis.operators import OPERATOR_APPLICATION_ORDER
    return OPERATOR_APPLICATION_ORDER


def operator_support_policies() -> dict[str, str]:
    from prism_fas.synthesis.operators import OPERATOR_SUPPORT_POLICIES
    return dict(OPERATOR_SUPPORT_POLICIES)


def derive_seed(bank_id: str, recipe_id: str, recipe_seed: int, sample_id: str, operator_index: int) -> int:
    """SHA256(bank_id | recipe_id | recipe_seed | sample_id | operator_index)
    folded into a bounded 32-bit integer. No global RNG is ever consulted."""
    material = "|".join((str(bank_id), str(recipe_id), str(int(recipe_seed)), str(sample_id), str(int(operator_index))))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def local_rng(seed: int) -> np.random.Generator:
    """A local, explicitly seeded generator. Never `np.random.*` module state."""
    return np.random.Generator(np.random.PCG64(int(seed) % SEED_MODULUS))


@dataclass(frozen=True)
class CompiledOperatorNode:
    node_index: int
    operator_name: str
    strength: float
    parameters: dict[str, float]
    support_policy: str
    seed: int
    input_key: str
    output_key: str

    def as_dict(self) -> dict[str, Any]:
        return {"node_index": self.node_index, "operator_name": self.operator_name,
                "strength": round(float(self.strength), 6), "parameters": {k: round(float(v), 6) for k, v in sorted(self.parameters.items())},
                "support_policy": self.support_policy, "seed": int(self.seed),
                "input_key": self.input_key, "output_key": self.output_key}


@dataclass(frozen=True)
class CompiledRecipeGraph:
    graph_schema_version: str
    compiler_version: str
    bank_id: str
    recipe_id: str
    recipe_hash: str
    ontology_version: str
    ontology_sha256: str
    recipe_seed: int
    requested_regions: tuple[str, ...]
    region_mask_policy: dict[str, Any]
    nodes: tuple[CompiledOperatorNode, ...]
    capture: dict[str, Any]
    generator_routes: tuple[str, ...]
    conditioning_version: str
    conditioning_dimension: int
    conditioning_vector: tuple[float, ...]
    conditioning_feature_names: tuple[str, ...]
    conditioning_feature_names_sha256: str
    graph_hash: str = field(default="")

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "graph_schema_version": self.graph_schema_version, "compiler_version": self.compiler_version,
            "bank_id": self.bank_id, "recipe_id": self.recipe_id, "recipe_hash": self.recipe_hash,
            "ontology_version": self.ontology_version, "ontology_sha256": self.ontology_sha256,
            "recipe_seed": int(self.recipe_seed), "requested_regions": list(self.requested_regions),
            "region_mask_policy": self.region_mask_policy, "nodes": [node.as_dict() for node in self.nodes],
            "capture": self.capture, "generator_routes": list(self.generator_routes),
            "conditioning_version": self.conditioning_version, "conditioning_dimension": self.conditioning_dimension,
            "conditioning_vector": [round(float(value), 6) for value in self.conditioning_vector],
            "conditioning_feature_names": list(self.conditioning_feature_names),
            "conditioning_feature_names_sha256": self.conditioning_feature_names_sha256}
        if include_hash: body["graph_hash"] = self.graph_hash
        return body

    def canonical_json(self, *, include_hash: bool = True) -> str:
        return json.dumps(self.payload(include_hash=include_hash), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def operator_names(self) -> tuple[str, ...]:
        return tuple(node.operator_name for node in self.nodes)

    def node_seed(self, node: CompiledOperatorNode, sample_id: str) -> int:
        """Per-sample operator seed, re-derived from the same formula that
        produced the compile-time seed with an empty sample scope."""
        return derive_seed(self.bank_id, self.recipe_id, self.recipe_seed, sample_id, node.node_index)

    def conditioning_array(self) -> np.ndarray:
        return np.asarray(self.conditioning_vector, dtype=np.float32)


def _resolve_parameters(recipe: RecipeV11, ontology: Ontology, name: str, strength: float, seed: int) -> dict[str, float]:
    """Deterministic compile-time operator parameters.

    Explicit recipe `parameters` win; anything the recipe leaves out is drawn
    from a node-local RNG inside the documented ontology band, so the value is
    fixed by the graph and recorded in the graph hash.
    """
    spec = recipe.artifact(name)
    explicit = dict(spec.parameters) if spec is not None else {}
    rng = local_rng(seed)
    resolved: dict[str, float] = {}
    for key in ontology.parameter_keys(name):
        band = ontology.parameter_range(name, key)
        if band is None: continue
        if key in explicit:
            value = float(explicit[key])
            if not band.contains(value): raise CompileError(f"{recipe.recipe_id}: {name}.{key}={value} outside [{band.minimum}, {band.maximum}]")
        else:
            value = float(band.minimum + (band.maximum - band.minimum) * float(rng.random()))
        resolved[key] = round(value, 6)
    unknown = sorted(set(explicit) - set(resolved))
    if unknown: raise CompileError(f"{recipe.recipe_id}: {name} carries undocumented parameters {unknown}")
    # capture-driven, strength-driven terms are part of the frozen graph too
    resolved["capture_motion"] = round(float(recipe.capture.motion), 6)
    resolved["capture_defocus"] = round(float(recipe.capture.defocus), 6)
    resolved["capture_scale"] = round(float(recipe.capture.scale), 6)
    resolved["capture_yaw"] = round(float(recipe.capture.yaw), 6)
    resolved["medium_transparency"] = round(float(recipe.medium.transparency), 6)
    resolved["medium_roughness"] = round(float(recipe.medium.roughness), 6)
    resolved["geometry_rigidity"] = round(float(recipe.geometry.rigidity), 6)
    resolved["strength"] = round(float(strength), 6)
    return resolved


def compile_recipe(recipe: RecipeV11, ontology: Ontology, *, bank_id: str,
                   coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE) -> CompiledRecipeGraph:
    """Deterministically compile one recipe into an operator graph.

    No image tensor is required. Identical ontology + recipe + compiler version
    always produce byte-identical canonical graph JSON and an identical hash.
    """
    if "physics" not in recipe.generator_route:
        raise CompileError(f"{recipe.recipe_id}: the physics route is required to compile an operator graph")
    order = operator_application_order()
    policies = operator_support_policies()
    requested = [spec.name for spec in recipe.artifacts]
    unknown = [name for name in requested if name not in order]
    if unknown: raise CompileError(f"{recipe.recipe_id}: no implemented physics operator for {unknown}")
    ordered = [name for name in order if name in requested]
    nodes: list[CompiledOperatorNode] = []
    for index, name in enumerate(ordered):
        spec = recipe.artifact(name)
        if spec is None: raise CompileError(f"{recipe.recipe_id}: artifact {name!r} disappeared during compilation")
        seed = derive_seed(bank_id, recipe.recipe_id, recipe.seed, "", index)
        nodes.append(CompiledOperatorNode(
            node_index=index, operator_name=name, strength=round(float(spec.strength), 6),
            parameters=_resolve_parameters(recipe, ontology, name, spec.strength, seed),
            support_policy=policies.get(name, "requested_region"), seed=seed,
            input_key="image" if index == 0 else f"image@{index}", output_key=f"image@{index + 1}"))
    vector = conditioning_vector(recipe, ontology)
    graph = CompiledRecipeGraph(
        graph_schema_version=GRAPH_SCHEMA_VERSION, compiler_version=COMPILER_VERSION, bank_id=bank_id,
        recipe_id=recipe.recipe_id, recipe_hash=recipe_hash(recipe), ontology_version=ontology.version,
        ontology_sha256=ontology.sha256, recipe_seed=int(recipe.seed), requested_regions=tuple(recipe.regions),
        region_mask_policy={"policy": MASK_POLICY, "regions": list(recipe.regions),
                            "geometry_shape": recipe.geometry.shape,
                            "requested_coverage": round(float(recipe.geometry.coverage), 6),
                            "coverage_tolerance": round(float(coverage_tolerance), 6)},
        nodes=tuple(nodes), capture=json.loads(canonical_json(recipe))["capture"],
        generator_routes=tuple(recipe.generator_route), conditioning_version=CONDITIONING_VERSION,
        conditioning_dimension=CONDITIONING_DIM, conditioning_vector=tuple(float(value) for value in vector),
        conditioning_feature_names=feature_names(ontology),
        conditioning_feature_names_sha256=feature_names_sha256(ontology))
    digest = hashlib.sha256(graph.canonical_json(include_hash=False).encode("utf-8")).hexdigest()
    return replace(graph, graph_hash=digest)


def compile_recipes(recipes: list[RecipeV11], ontology: Ontology, *, bank_id: str) -> list[CompiledRecipeGraph]:
    return [compile_recipe(recipe, ontology, bank_id=bank_id) for recipe in recipes]


def compile_summary(graphs: list[CompiledRecipeGraph]) -> dict[str, Any]:
    hashes = [graph.graph_hash for graph in graphs]
    return {"compiler_version": COMPILER_VERSION, "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "compiled": len(graphs), "unique_graph_hashes": len(set(hashes)),
            "duplicate_graph_hashes": sorted({value for value in hashes if hashes.count(value) > 1}),
            "conditioning_version": CONDITIONING_VERSION, "conditioning_dimension": CONDITIONING_DIM,
            "operator_application_order": list(operator_application_order()),
            "graph_hashes": {graph.recipe_id: graph.graph_hash for graph in graphs},
            "graph_hash_set_sha256": stable_hash(sorted(hashes))}
