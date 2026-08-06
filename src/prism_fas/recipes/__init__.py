"""M7 source-only recipe schema, ontology, validation, compiler and frozen bank.

Nothing in this package reads target data, source_dev data or any dataset path:
a recipe is a dataset-agnostic description of presentation physics.
"""
from __future__ import annotations
from .canonical import canonical_json, recipe_description, recipe_hash
from .compile import (COMPILER_VERSION, CompiledOperatorNode, CompiledRecipeGraph, compile_recipe, compile_recipes,
                      compile_summary, derive_seed, local_rng)
from .conditioning import (CONDITIONING_DIM, CONDITIONING_VERSION, conditioning_vector, decode_conditioning,
                           feature_names, feature_names_sha256)
from .ontology import Ontology, OntologyError, load_ontology, parse_ontology
from .schema import RECIPE_SCHEMA_VERSION, RecipeSchemaError, RecipeV11, parse_recipe
from .validate import RecipeValidationError, ValidationIssue, validate_payload, validate_recipe, validate_recipes

__all__ = ["COMPILER_VERSION", "CONDITIONING_DIM", "CONDITIONING_VERSION", "CompiledOperatorNode",
           "CompiledRecipeGraph", "Ontology", "OntologyError", "RECIPE_SCHEMA_VERSION", "RecipeSchemaError",
           "RecipeV11", "RecipeValidationError", "ValidationIssue", "canonical_json", "compile_recipe",
           "compile_recipes", "compile_summary", "conditioning_vector", "decode_conditioning", "derive_seed",
           "feature_names", "feature_names_sha256", "load_ontology", "local_rng", "parse_ontology", "parse_recipe",
           "recipe_description", "recipe_hash", "validate_payload", "validate_recipe", "validate_recipes"]
