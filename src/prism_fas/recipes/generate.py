from __future__ import annotations
import json
from typing import Any
import numpy as np
from .canonical import canonical_json, recipe_description, recipe_hash
from .compile import derive_seed, local_rng
from .ontology import Ontology
from .schema import RecipeV11, parse_recipe

# The M7 bank is produced by an offline deterministic structured generator.
# No external API, credential or hosted model is involved; `prompt.txt` is the
# constrained contract a future frozen LLM provider would be given instead.
GENERATOR_PROVIDER = "deterministic_local"
GENERATOR_MODEL_ID = "deterministic-source-only-recipe-generator"
GENERATOR_REVISION = "m7-v1"
GENERATOR_EXTERNAL_LLM_INVOKED = False


class GenerationError(ValueError):
    """The deterministic generator could not satisfy the required coverage."""


def _rotate(values: tuple[str, ...] | list[str], offset: int, count: int) -> list[str]:
    items = list(values)
    if not items: raise GenerationError("cannot rotate an empty vocabulary")
    count = min(count, len(items))
    start = int(offset) % len(items)
    return [items[(start + step) % len(items)] for step in range(count)]


def _round(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals) + 0.0


def _allocate_strengths(names: list[str], ontology: Ontology, rng: np.random.Generator) -> list[float]:
    """Deterministic severity allocation inside the per-artifact safe bands and
    under the ontology's total-severity budget."""
    budget = float(ontology.limits["max_total_artifact_strength"])
    bands = [ontology.strength_range(name) for name in names]
    total = float(rng.uniform(0.20, min(0.95, budget)))
    weights = np.asarray(rng.random(len(names)), dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-9)
    values = [_round(min(max(total * float(weight), band.minimum), band.maximum)) for weight, band in zip(weights, bands)]
    guard = 0
    while sum(values) > budget + 1e-9:
        guard += 1
        if guard > 64: raise GenerationError(f"could not fit {names} inside the severity budget {budget}")
        excess = sum(values) - budget
        index = max(range(len(values)), key=lambda position: values[position] - bands[position].minimum)
        headroom = values[index] - bands[index].minimum
        if headroom <= 1e-9: raise GenerationError(f"severity budget {budget} unreachable for {names}")
        values[index] = _round(values[index] - min(excess, headroom))
    return values


def build_recipe(index: int, ontology: Ontology, *, bank_id: str, bank_seed: int) -> RecipeV11:
    """One deterministic recipe.

    Categorical axes are advanced by fixed strides so that the required coverage
    is a property of the construction; continuous values come from a recipe-local
    RNG seeded from (bank_id, recipe_id, bank_seed) via the same SHA-256
    derivation the compiler uses. No global RNG is touched.
    """
    recipe_id = f"R-{index + 1:06d}"
    rng = local_rng(derive_seed(bank_id, recipe_id, bank_seed, "generator", index))
    family = ontology.media[index % len(ontology.media)]
    shape = ontology.geometry_shapes[(index // len(ontology.media)) % len(ontology.geometry_shapes)]
    illumination = ontology.illumination[(index // 3) % len(ontology.illumination)]
    max_artifacts = int(ontology.limits["max_artifacts_per_recipe"])
    max_regions = int(ontology.limits["max_regions_per_recipe"])
    artifact_count = 1 + (index % max_artifacts)
    region_count = 1 + ((index // 7) % max_regions)
    names = _rotate(ontology.artifacts_for_medium(family), index // 2, artifact_count)
    regions = ontology.sort_regions(_rotate(ontology.regions_for_geometry(shape), index // 4, region_count))
    strengths = _allocate_strengths(names, ontology, rng)
    transparency = ontology.medium_ranges["transparency"]
    roughness = ontology.medium_ranges["roughness"]
    rigidity = ontology.geometry_ranges["rigidity"]
    coverage = ontology.geometry_ranges["coverage"]
    yaw = ontology.capture_ranges["yaw"]
    compression = ontology.capture_ranges["compression_q"]
    scale = ontology.capture_ranges["scale"]
    motion = ontology.capture_ranges["motion"]
    defocus = ontology.capture_ranges["defocus"]
    shortcuts = [name for name in ontology.forbidden_shortcuts
                 if names != [{"always_moire": "moire", "always_halftone": "halftone"}.get(name)]]
    routes = ["physics", "gpat"] if index % 2 == 0 else ["physics"]
    payload: dict[str, Any] = {
        "recipe_id": recipe_id,
        "medium": {"family": family,
                   "transparency": _round(rng.uniform(transparency.minimum, transparency.maximum)),
                   "roughness": _round(rng.uniform(roughness.minimum, roughness.maximum))},
        "geometry": {"shape": shape,
                     "rigidity": _round(rng.uniform(rigidity.minimum, rigidity.maximum)),
                     "coverage": _round(rng.uniform(coverage.minimum, coverage.maximum))},
        "regions": regions,
        "artifacts": [{"name": name, "strength": strength} for name, strength in zip(names, strengths)],
        "capture": {"yaw": _round(rng.uniform(yaw.minimum, yaw.maximum), 1),
                    "illumination": illumination,
                    "compression_q": int(rng.integers(int(compression.minimum), int(compression.maximum) + 1)),
                    "scale": _round(rng.uniform(scale.minimum, scale.maximum), 3),
                    "motion": _round(rng.uniform(motion.minimum, motion.maximum), 3),
                    "defocus": _round(rng.uniform(defocus.minimum, defocus.maximum), 3)},
        "forbidden_shortcuts": list(shortcuts),
        "generator_route": routes,
        "seed": int(rng.integers(int(ontology.limits["seed_min"]), int(ontology.limits["seed_max"]))),
        "schema_version": ontology.recipe_schema_version}
    return parse_recipe(payload)


def generate_recipes(ontology: Ontology, *, count: int, bank_id: str, bank_seed: int) -> list[RecipeV11]:
    recipes = [build_recipe(index, ontology, bank_id=bank_id, bank_seed=bank_seed) for index in range(int(count))]
    digests = [recipe_hash(recipe) for recipe in recipes]
    duplicates = sorted({value for value in digests if digests.count(value) > 1})
    if duplicates: raise GenerationError(f"deterministic generator produced duplicate canonical hashes: {duplicates}")
    missing = _coverage_gaps(recipes, ontology)
    if missing: raise GenerationError(f"generated bank does not meet required categorical coverage: {missing}")
    return recipes


def _coverage_gaps(recipes: list[RecipeV11], ontology: Ontology) -> dict[str, list[str]]:
    seen_media = {recipe.medium.family for recipe in recipes}
    seen_geometry = {recipe.geometry.shape for recipe in recipes}
    seen_regions = {name for recipe in recipes for name in recipe.regions}
    seen_artifacts = {spec.name for recipe in recipes for spec in recipe.artifacts}
    seen_illumination = {recipe.capture.illumination for recipe in recipes}
    gaps: dict[str, list[str]] = {}
    for key, expected, seen in (("media", ontology.media, seen_media), ("geometry_shapes", ontology.geometry_shapes, seen_geometry),
                                ("regions", ontology.regions, seen_regions), ("artifacts", ontology.artifacts, seen_artifacts),
                                ("illumination", ontology.illumination, seen_illumination)):
        absent = [value for value in expected if value not in seen]
        if absent: gaps[key] = absent
    if not any("physics" in recipe.generator_route for recipe in recipes): gaps["routes"] = ["physics"]
    if any("physics" not in recipe.generator_route for recipe in recipes): gaps["physics_route_missing"] = ["every recipe must declare physics"]
    counts = {len(recipe.artifacts) for recipe in recipes}
    absent_counts = [str(value) for value in (1, 2, 3) if value not in counts]
    if absent_counts: gaps["artifact_counts"] = absent_counts
    region_counts = {len(recipe.regions) for recipe in recipes}
    if not any(value == 1 for value in region_counts): gaps["region_counts"] = ["single-region recipe missing"]
    if not any(value > 1 for value in region_counts): gaps.setdefault("region_counts", []).append("multi-region recipe missing")
    return gaps


def render_prompt(ontology: Ontology) -> str:
    """The constrained structured recipe-generation contract.

    It is frozen alongside the bank so a future provider can be held to exactly
    the same schema, vocabulary and source-only constraints. M7 does not send it
    anywhere: the bank in this repository was produced by the deterministic
    local generator above.
    """
    example = build_recipe(0, ontology, bank_id="prism_recipe_bank_m7_v1", bank_seed=20260806)
    lines = [
        "PRISM-FAS-B source-only attack recipe generation contract",
        f"ontology_version: {ontology.version}",
        f"ontology_sha256: {ontology.sha256}",
        f"recipe_schema_version: {ontology.recipe_schema_version}",
        "",
        "ROLE",
        "Emit structured presentation-attack recipes describing general physical",
        "properties of a re-presented face surface. Emit JSON only.",
        "",
        "HARD CONSTRAINTS",
        "1. One JSON object per line. No prose, no markdown, no code fence.",
        "2. Exactly these top-level keys, no others: recipe_id, medium, geometry,",
        "   regions, artifacts, capture, forbidden_shortcuts, generator_route, seed,",
        "   schema_version.",
        f"3. recipe_id matches {ontology.limits['recipe_id_pattern']} and is unique.",
        f"4. schema_version is exactly \"{ontology.recipe_schema_version}\".",
        f"5. seed is an integer in [{ontology.limits['seed_min']}, {ontology.limits['seed_max']}].",
        "6. You never see, name or infer any evaluation/target dataset, attack",
        "   taxonomy, file path, subject identity or model performance number.",
        "   Recipes are dataset-agnostic descriptions of physics.",
        f"7. At most {ontology.limits['max_artifacts_per_recipe']} artifacts and "
        f"{ontology.limits['max_regions_per_recipe']} regions per recipe; artifact names and",
        "   region names are unique within a recipe; regions use the canonical order below.",
        f"8. The sum of artifact strengths never exceeds {ontology.limits['max_total_artifact_strength']}.",
        "9. Every artifact must be compatible with the chosen medium and every region",
        "   compatible with the chosen geometry (tables below).",
        "10. generator_route is a non-empty unique subset of "
        f"{list(ontology.routes)} and must contain \"physics\".",
        "",
        "VOCABULARY",
        f"medium.family: {list(ontology.media)}",
        f"geometry.shape: {list(ontology.geometry_shapes)}",
        f"regions (canonical order): {list(ontology.regions)}",
        f"artifacts[].name: {list(ontology.artifacts)}",
        f"capture.illumination: {list(ontology.illumination)}",
        f"forbidden_shortcuts: {list(ontology.forbidden_shortcuts)}",
        "",
        "RANGES",
        f"medium.transparency {ontology.medium_ranges['transparency'].as_dict()}",
        f"medium.roughness {ontology.medium_ranges['roughness'].as_dict()}",
        f"geometry.rigidity {ontology.geometry_ranges['rigidity'].as_dict()}",
        f"geometry.coverage {ontology.geometry_ranges['coverage'].as_dict()}",
        f"capture.yaw {ontology.capture_ranges['yaw'].as_dict()}",
        f"capture.compression_q {ontology.capture_ranges['compression_q'].as_dict()} (integer)",
        f"capture.scale {ontology.capture_ranges['scale'].as_dict()}",
        f"capture.motion {ontology.capture_ranges['motion'].as_dict()}",
        f"capture.defocus {ontology.capture_ranges['defocus'].as_dict()}",
        "artifact strength safe bands:"]
    for name in ontology.artifacts:
        lines.append(f"  {name}: {ontology.strength_range(name).as_dict()}")
    lines += ["", "MEDIUM / ARTIFACT COMPATIBILITY"]
    for family in ontology.media:
        lines.append(f"  {family}: {list(ontology.artifacts_for_medium(family))}")
    lines += ["", "GEOMETRY / REGION COMPATIBILITY"]
    for shape in ontology.geometry_shapes:
        lines.append(f"  {shape}: {list(ontology.regions_for_geometry(shape))}")
    lines += ["", "EXAMPLE OUTPUT LINE", canonical_json(example), "",
              "EXAMPLE PLAIN DESCRIPTION (for reference only, never emit prose)",
              recipe_description(example), ""]
    return "\n".join(lines)


def generator_manifest(*, bank_id: str, bank_seed: int, count: int, ontology: Ontology) -> dict[str, Any]:
    return {"provider": GENERATOR_PROVIDER, "model_id": GENERATOR_MODEL_ID, "revision": GENERATOR_REVISION,
            "external_llm_invoked": GENERATOR_EXTERNAL_LLM_INVOKED, "network_access": False, "credential_used": False,
            "bank_id": bank_id, "bank_seed": int(bank_seed), "requested_recipe_count": int(count),
            "ontology_version": ontology.version, "ontology_sha256": ontology.sha256,
            "recipe_schema_version": ontology.recipe_schema_version,
            "policy": "offline deterministic structured generation; prompt.txt is the frozen contract a future "
                      "pinned LLM provider would receive, and was not sent anywhere in M7",
            "seed_derivation": "sha256(bank_id|recipe_id|bank_seed|'generator'|index) -> PCG64"}


def recipes_to_jsonl(recipes: list[RecipeV11]) -> str:
    return "".join(f"{canonical_json(recipe)}\n" for recipe in recipes)


def recipes_from_jsonl(text: str) -> list[RecipeV11]:
    recipes: list[RecipeV11] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        try: payload = json.loads(line)
        except json.JSONDecodeError as exc: raise GenerationError(f"recipes.jsonl line {number} is not valid JSON: {exc}") from exc
        recipes.append(parse_recipe(payload))
    return recipes
