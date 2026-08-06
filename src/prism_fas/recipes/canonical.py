from __future__ import annotations
import hashlib, json
from typing import Any
from .schema import RecipeV11

CANONICAL_FLOAT_DECIMALS = 6


def _round(value: Any) -> Any:
    """Round every float to a fixed number of decimals so that the canonical
    text (and therefore the hash) cannot depend on float repr differences."""
    if isinstance(value, bool): return value
    if isinstance(value, float): return round(value, CANONICAL_FLOAT_DECIMALS) + 0.0
    if isinstance(value, dict): return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_round(item) for item in value]
    return value


def canonical_payload(recipe: RecipeV11) -> dict[str, Any]:
    return _round(recipe.model_dump(mode="json"))


def canonical_json(recipe: RecipeV11) -> str:
    """Deterministic single-line JSON: sorted keys, no whitespace padding."""
    return json.dumps(canonical_payload(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def recipe_hash(recipe: RecipeV11) -> str:
    return hashlib.sha256(canonical_json(recipe).encode("utf-8")).hexdigest()


def recipe_description(recipe: RecipeV11) -> str:
    """Deterministic natural-language description of one recipe.

    Used by the offline diversity audit and by `prompt.txt` examples. It carries
    only recipe content: no dataset name, path, sample id or target information.
    """
    artifacts = " ".join(f"{spec.name} at strength {spec.strength:.2f}" for spec in recipe.artifacts)
    return (f"{recipe.medium.family} medium with transparency {recipe.medium.transparency:.2f} and roughness "
            f"{recipe.medium.roughness:.2f} in {recipe.geometry.shape} geometry with rigidity "
            f"{recipe.geometry.rigidity:.2f} covering {recipe.geometry.coverage:.2f} of regions "
            f"{' '.join(recipe.regions)} showing {artifacts} captured at yaw {recipe.capture.yaw:.1f} with "
            f"{recipe.capture.illumination} illumination compression {recipe.capture.compression_q} scale "
            f"{recipe.capture.scale:.2f} motion {recipe.capture.motion:.2f} defocus {recipe.capture.defocus:.2f} "
            f"via {' '.join(recipe.generator_route)}")


def canonical_lines(recipes: list[RecipeV11]) -> str:
    """The exact `recipes.jsonl` body: one canonical recipe per line, LF only."""
    return "".join(f"{canonical_json(recipe)}\n" for recipe in recipes)


def stable_hash(value: Any) -> str:
    raw = json.dumps(_round(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
