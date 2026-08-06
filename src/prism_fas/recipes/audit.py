from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any
import numpy as np
from .canonical import recipe_description, recipe_hash, stable_hash
from .conditioning import conditioning_vector
from .ontology import Ontology
from .schema import RecipeV11

# Offline, dependency-free diversity method. No text model is downloaded and no
# network call is made: the vocabulary is built from the deterministic recipe
# descriptions themselves.
DIVERSITY_METHOD = "offline_tfidf_cosine_v1"
DIVERSITY_MAX_PAIRWISE_COSINE = 0.98
DIVERSITY_MEAN_PAIRWISE_COSINE_MAX = 0.90
_TOKEN = re.compile(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def tfidf_matrix(documents: list[str]) -> tuple[np.ndarray, list[str]]:
    """L2-normalized TF-IDF rows with smoothed IDF, computed locally."""
    tokenized = [tokenize(document) for document in documents]
    vocabulary = sorted({token for tokens in tokenized for token in tokens})
    index = {token: position for position, token in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(tokenized):
        for token, count in Counter(tokens).items():
            counts[row, index[token]] = float(count)
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
    matrix = counts * idf[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, vocabulary


def _pairwise_stats(matrix: np.ndarray) -> dict[str, float]:
    similarity = matrix @ matrix.T
    rows = similarity.shape[0]
    if rows < 2: return {"pairs": 0, "mean_cosine": 0.0, "max_cosine": 0.0, "min_cosine": 0.0}
    upper = similarity[np.triu_indices(rows, k=1)]
    return {"pairs": int(upper.size), "mean_cosine": float(upper.mean()), "max_cosine": float(upper.max()),
            "min_cosine": float(upper.min())}


def coverage_table(recipes: list[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    media = Counter(recipe.medium.family for recipe in recipes)
    geometry = Counter(recipe.geometry.shape for recipe in recipes)
    regions = Counter(name for recipe in recipes for name in recipe.regions)
    artifacts = Counter(spec.name for recipe in recipes for spec in recipe.artifacts)
    illumination = Counter(recipe.capture.illumination for recipe in recipes)
    routes = Counter(route for recipe in recipes for route in recipe.generator_route)
    artifact_counts = Counter(len(recipe.artifacts) for recipe in recipes)
    region_counts = Counter(len(recipe.regions) for recipe in recipes)
    required = {
        "every_medium_present": all(media.get(value, 0) > 0 for value in ontology.media),
        "every_geometry_present": all(geometry.get(value, 0) > 0 for value in ontology.geometry_shapes),
        "every_region_present": all(regions.get(value, 0) > 0 for value in ontology.regions),
        "every_artifact_present": all(artifacts.get(value, 0) > 0 for value in ontology.artifacts),
        "every_illumination_present": all(illumination.get(value, 0) > 0 for value in ontology.illumination),
        "physics_route_in_every_recipe": all("physics" in recipe.generator_route for recipe in recipes),
        "single_region_recipes_present": region_counts.get(1, 0) > 0,
        "multi_region_recipes_present": sum(count for size, count in region_counts.items() if size > 1) > 0,
        "artifact_count_1_present": artifact_counts.get(1, 0) > 0,
        "artifact_count_2_present": artifact_counts.get(2, 0) > 0,
        "artifact_count_3_present": artifact_counts.get(3, 0) > 0}
    return {"recipe_count": len(recipes),
            "media": {value: int(media.get(value, 0)) for value in ontology.media},
            "geometry_shapes": {value: int(geometry.get(value, 0)) for value in ontology.geometry_shapes},
            "regions": {value: int(regions.get(value, 0)) for value in ontology.regions},
            "artifacts": {value: int(artifacts.get(value, 0)) for value in ontology.artifacts},
            "illumination": {value: int(illumination.get(value, 0)) for value in ontology.illumination},
            "routes": {value: int(routes.get(value, 0)) for value in ontology.routes},
            "artifacts_per_recipe": {str(size): int(count) for size, count in sorted(artifact_counts.items())},
            "regions_per_recipe": {str(size): int(count) for size, count in sorted(region_counts.items())},
            "required": required, "required_all_met": all(required.values())}


def diversity_audit(recipes: list[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    descriptions = [recipe_description(recipe) for recipe in recipes]
    matrix, vocabulary = tfidf_matrix(descriptions)
    text_stats = _pairwise_stats(matrix)
    vectors = np.stack([conditioning_vector(recipe, ontology).astype(np.float64) for recipe in recipes])
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    structured_stats = _pairwise_stats(normalized)
    digests = [recipe_hash(recipe) for recipe in recipes]
    duplicate_descriptions = sorted({value for value in descriptions if descriptions.count(value) > 1})
    passed = (text_stats["max_cosine"] <= DIVERSITY_MAX_PAIRWISE_COSINE
              and text_stats["mean_cosine"] <= DIVERSITY_MEAN_PAIRWISE_COSINE_MAX
              and not duplicate_descriptions
              and len(set(digests)) == len(digests))
    return {"method": DIVERSITY_METHOD, "external_text_model": False, "network_access": False,
            "vocabulary_size": len(vocabulary), "documents": len(descriptions),
            "thresholds": {"max_pairwise_cosine": DIVERSITY_MAX_PAIRWISE_COSINE,
                           "mean_pairwise_cosine": DIVERSITY_MEAN_PAIRWISE_COSINE_MAX},
            "text_tfidf": {key: (round(value, 8) if isinstance(value, float) else value) for key, value in text_stats.items()},
            "structured_conditioning": {key: (round(value, 8) if isinstance(value, float) else value)
                                        for key, value in structured_stats.items()},
            "duplicate_descriptions": duplicate_descriptions,
            "duplicate_recipe_hashes": sorted({value for value in digests if digests.count(value) > 1}),
            "unique_recipe_hashes": len(set(digests)),
            "descriptions_sha256": stable_hash(descriptions), "passed": bool(passed)}


def bank_audit(recipes: list[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    coverage = coverage_table(recipes, ontology)
    diversity = diversity_audit(recipes, ontology)
    return {"coverage": coverage, "diversity": diversity,
            "passed": bool(coverage["required_all_met"] and diversity["passed"]),
            "entropy_bits": {
                "media": round(_entropy([recipe.medium.family for recipe in recipes]), 6),
                "geometry": round(_entropy([recipe.geometry.shape for recipe in recipes]), 6),
                "illumination": round(_entropy([recipe.capture.illumination for recipe in recipes]), 6)}}


def _entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total: return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
