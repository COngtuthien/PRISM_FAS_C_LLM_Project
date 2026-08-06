from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from .canonical import canonical_json, recipe_hash
from .ontology import Ontology
from .schema import RecipeSchemaError, RecipeV11, parse_recipe

VALIDATION_STAGES = ("schema", "canonical", "medium_artifact", "geometry_region", "strength_range", "severity",
                     "duplicate", "route", "shortcut", "leakage", "serialization", "hash")


class RecipeValidationError(ValueError):
    """One or more recipes failed validation. Never raised silently: the
    message carries the structured issues."""
    def __init__(self, issues: list["ValidationIssue"]):
        self.issues = issues
        super().__init__("; ".join(issue.text() for issue in issues))


@dataclass(frozen=True)
class ValidationIssue:
    recipe_id: str | None
    stage: str
    field: str
    reason: str
    def text(self) -> str: return f"[{self.recipe_id or '<unparsed>'}][{self.stage}][{self.field}] {self.reason}"
    def as_dict(self) -> dict[str, Any]: return asdict(self)


def physics_operator_names() -> tuple[str, ...]:
    """Operators the M7 physics engine actually implements. Imported lazily so
    the recipe package stays usable without the synthesis package loaded."""
    from prism_fas.synthesis.operators import OPERATOR_NAMES
    return OPERATOR_NAMES


def canonicalize_payload(payload: dict[str, Any], ontology: Ontology) -> dict[str, Any]:
    """Map documented aliases onto canonical values. Accepts the free-text form
    a future frozen LLM provider may emit; unknown keys are left untouched so
    the strict schema still rejects them."""
    if not isinstance(payload, dict): raise RecipeSchemaError("recipe payload must be a JSON object")
    result = dict(payload)
    medium = result.get("medium")
    if isinstance(medium, dict) and "family" in medium:
        result["medium"] = {**medium, "family": ontology.canonical_medium(medium["family"])}
    geometry = result.get("geometry")
    if isinstance(geometry, dict) and "shape" in geometry:
        result["geometry"] = {**geometry, "shape": ontology.canonical_geometry(geometry["shape"])}
    regions = result.get("regions")
    if isinstance(regions, list):
        result["regions"] = ontology.canonical_regions([str(name) for name in regions])
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        canonical: list[Any] = []
        for spec in artifacts:
            if isinstance(spec, dict) and "name" in spec:
                canonical.append({**spec, "name": ontology.canonical_artifact(spec["name"])})
            else: canonical.append(spec)
        result["artifacts"] = canonical
    capture = result.get("capture")
    if isinstance(capture, dict) and "illumination" in capture:
        result["capture"] = {**capture, "illumination": ontology.canonical_illumination(capture["illumination"])}
    return result


def _issue(recipe_id: str | None, stage: str, field: str, reason: str) -> ValidationIssue:
    return ValidationIssue(recipe_id=recipe_id, stage=stage, field=field, reason=reason)


def validate_recipe(recipe: RecipeV11, ontology: Ontology, *, operators: Iterable[str] | None = None) -> list[ValidationIssue]:
    """Every rule in one pass. Returns all issues rather than raising on the
    first, so an invalid bank reports every problem it has."""
    available = tuple(operators) if operators is not None else physics_operator_names()
    issues: list[ValidationIssue] = []
    rid = recipe.recipe_id
    limits = ontology.limits

    # canonical enum membership under this ontology version
    if recipe.medium.family not in ontology.media:
        issues.append(_issue(rid, "canonical", "medium.family", f"{recipe.medium.family!r} is not enabled by ontology {ontology.version}"))
    if recipe.geometry.shape not in ontology.geometry_shapes:
        issues.append(_issue(rid, "canonical", "geometry.shape", f"{recipe.geometry.shape!r} is not enabled by ontology {ontology.version}"))
    if recipe.capture.illumination not in ontology.illumination:
        issues.append(_issue(rid, "canonical", "capture.illumination", f"{recipe.capture.illumination!r} is not enabled by ontology {ontology.version}"))
    for name in recipe.regions:
        if name not in ontology.regions:
            issues.append(_issue(rid, "canonical", "regions", f"{name!r} is not enabled by ontology {ontology.version}"))
    if recipe.schema_version != ontology.recipe_schema_version:
        issues.append(_issue(rid, "canonical", "schema_version", f"{recipe.schema_version!r} != ontology recipe_schema_version {ontology.recipe_schema_version!r}"))

    # structural ranges owned by the ontology
    for field, value in (("medium.transparency", recipe.medium.transparency), ("medium.roughness", recipe.medium.roughness)):
        band = ontology.medium_ranges[field.split(".", 1)[1]]
        if not band.contains(value): issues.append(_issue(rid, "schema", field, f"{value} outside safe range [{band.minimum}, {band.maximum}]"))
    for field, value in (("geometry.rigidity", recipe.geometry.rigidity), ("geometry.coverage", recipe.geometry.coverage)):
        band = ontology.geometry_ranges[field.split(".", 1)[1]]
        if not band.contains(value): issues.append(_issue(rid, "schema", field, f"{value} outside safe range [{band.minimum}, {band.maximum}]"))
    for field, value in (("capture.yaw", recipe.capture.yaw), ("capture.compression_q", recipe.capture.compression_q),
                         ("capture.scale", recipe.capture.scale), ("capture.motion", recipe.capture.motion),
                         ("capture.defocus", recipe.capture.defocus)):
        band = ontology.capture_ranges[field.split(".", 1)[1]]
        if not band.contains(value): issues.append(_issue(rid, "schema", field, f"{value} outside safe range [{band.minimum}, {band.maximum}]"))

    # counts and duplicates
    if not (int(limits["min_regions_per_recipe"]) <= len(recipe.regions) <= int(limits["max_regions_per_recipe"])):
        issues.append(_issue(rid, "schema", "regions", f"{len(recipe.regions)} regions outside [{limits['min_regions_per_recipe']}, {limits['max_regions_per_recipe']}]"))
    if not (int(limits["min_artifacts_per_recipe"]) <= len(recipe.artifacts) <= int(limits["max_artifacts_per_recipe"])):
        issues.append(_issue(rid, "schema", "artifacts", f"{len(recipe.artifacts)} artifacts outside [{limits['min_artifacts_per_recipe']}, {limits['max_artifacts_per_recipe']}]"))
    if len(set(recipe.regions)) != len(recipe.regions):
        issues.append(_issue(rid, "duplicate", "regions", f"duplicate regions in {recipe.regions}"))
    names = [spec.name for spec in recipe.artifacts]
    if len(set(names)) != len(names):
        issues.append(_issue(rid, "duplicate", "artifacts", f"duplicate artifact names in {names}"))
    if recipe.regions != ontology.sort_regions(list(recipe.regions)):
        issues.append(_issue(rid, "canonical", "regions", f"regions are not in canonical order: expected {ontology.sort_regions(list(recipe.regions))}"))
    if not (int(limits["seed_min"]) <= recipe.seed <= int(limits["seed_max"])):
        issues.append(_issue(rid, "schema", "seed", f"seed {recipe.seed} outside [{limits['seed_min']}, {limits['seed_max']}]"))

    # medium / artifact compatibility
    try: allowed_artifacts = ontology.artifacts_for_medium(recipe.medium.family)
    except Exception: allowed_artifacts = ()
    for spec in recipe.artifacts:
        if allowed_artifacts and spec.name not in allowed_artifacts:
            issues.append(_issue(rid, "medium_artifact", "artifacts",
                                 f"medium {recipe.medium.family!r} cannot produce {spec.name!r}; allowed {list(allowed_artifacts)}"))

    # geometry / region compatibility
    try: allowed_regions = ontology.regions_for_geometry(recipe.geometry.shape)
    except Exception: allowed_regions = ()
    for name in recipe.regions:
        if allowed_regions and name not in allowed_regions:
            issues.append(_issue(rid, "geometry_region", "regions",
                                 f"geometry {recipe.geometry.shape!r} cannot cover {name!r}; allowed {list(allowed_regions)}"))

    # safe strength band and severity budget
    total = 0.0
    for spec in recipe.artifacts:
        total += float(spec.strength)
        try: band = ontology.strength_range(spec.name)
        except Exception: continue
        if not band.contains(spec.strength):
            issues.append(_issue(rid, "strength_range", f"artifacts.{spec.name}.strength",
                                 f"{spec.strength} outside operator-safe range [{band.minimum}, {band.maximum}]"))
        for key, value in sorted(spec.parameters.items()):
            band = ontology.parameter_range(spec.name, key)
            if band is None:
                issues.append(_issue(rid, "schema", f"artifacts.{spec.name}.parameters.{key}",
                                     "parameter is not documented in the ontology"))
            elif not band.contains(value):
                issues.append(_issue(rid, "strength_range", f"artifacts.{spec.name}.parameters.{key}",
                                     f"{value} outside documented range [{band.minimum}, {band.maximum}]"))
    budget = float(limits["max_total_artifact_strength"])
    if total > budget + 1e-9:
        issues.append(_issue(rid, "severity", "artifacts", f"total artifact strength {total:.6f} exceeds budget {budget}"))

    # routes: every declared route must be enabled, physics must be implemented
    for route in recipe.generator_route:
        if route not in ontology.routes:
            issues.append(_issue(rid, "route", "generator_route", f"route {route!r} is not enabled by the ontology"))
    if "physics" in recipe.generator_route:
        missing = [spec.name for spec in recipe.artifacts if spec.name not in available]
        if missing:
            issues.append(_issue(rid, "route", "generator_route",
                                 f"physics route requires implemented operators; missing {missing}"))

    # forbidden-shortcut consistency
    for shortcut in recipe.forbidden_shortcuts:
        if shortcut not in ontology.forbidden_shortcuts:
            issues.append(_issue(rid, "shortcut", "forbidden_shortcuts", f"{shortcut!r} is not an ontology shortcut policy"))
        artifact = {"always_moire": "moire", "always_halftone": "halftone"}.get(shortcut)
        if artifact and names == [artifact]:
            issues.append(_issue(rid, "shortcut", "forbidden_shortcuts",
                                 f"recipe declares {shortcut!r} forbidden yet {artifact!r} is its only artifact"))

    # source-only leakage scan over the exact canonical text
    try: text = canonical_json(recipe)
    except Exception as exc:
        issues.append(_issue(rid, "serialization", "recipe", f"canonical serialization failed: {exc}")); text = ""
    if text:
        hits = ontology.leakage_hits(text)
        if hits: issues.append(_issue(rid, "leakage", "recipe", f"forbidden source-only tokens present: {hits}"))
        try:
            replayed = parse_recipe(json.loads(text))
            if canonical_json(replayed) != text:
                issues.append(_issue(rid, "serialization", "recipe", "canonical JSON is not a fixed point of parse/serialize"))
            if recipe_hash(replayed) != recipe_hash(recipe):
                issues.append(_issue(rid, "hash", "recipe", "content hash is not stable across a serialize/parse round trip"))
        except Exception as exc:
            issues.append(_issue(rid, "serialization", "recipe", f"canonical JSON does not re-parse: {exc}"))
    return issues


def validate_payload(payload: dict[str, Any], ontology: Ontology, *, canonicalize: bool = False,
                     operators: Iterable[str] | None = None) -> tuple[RecipeV11 | None, list[ValidationIssue]]:
    """Parse then validate. A schema failure returns `(None, [issue])` — the
    payload is never partially accepted and no field is silently dropped."""
    rid = payload.get("recipe_id") if isinstance(payload, dict) else None
    try:
        prepared = canonicalize_payload(payload, ontology) if canonicalize else payload
        recipe = parse_recipe(prepared)
    except Exception as exc:
        return None, [_issue(rid if isinstance(rid, str) else None, "schema", "recipe", str(exc))]
    return recipe, validate_recipe(recipe, ontology, operators=operators)


def validate_recipes(recipes: list[RecipeV11], ontology: Ontology, *, operators: Iterable[str] | None = None) -> dict[str, Any]:
    """Bank-level validation: per-recipe rules plus uniqueness of ids/hashes and
    the bank-wide `forbidden_shortcuts` policy."""
    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, str] = {}
    for recipe in recipes:
        issues.extend(validate_recipe(recipe, ontology, operators=operators))
        if recipe.recipe_id in seen_ids:
            issues.append(_issue(recipe.recipe_id, "duplicate", "recipe_id", "recipe id appears more than once in the bank"))
        seen_ids[recipe.recipe_id] = seen_ids.get(recipe.recipe_id, 0) + 1
        digest = recipe_hash(recipe)
        if digest in seen_hashes:
            issues.append(_issue(recipe.recipe_id, "duplicate", "recipe_hash",
                                 f"canonical hash collides with {seen_hashes[digest]}"))
        else: seen_hashes[digest] = recipe.recipe_id
    # A shortcut is "always X" only if every recipe in the bank carries X.
    if recipes:
        for shortcut, artifact in (("always_moire", "moire"), ("always_halftone", "halftone")):
            declared = [recipe for recipe in recipes if shortcut in recipe.forbidden_shortcuts]
            if declared and all(any(spec.name == artifact for spec in recipe.artifacts) for recipe in recipes):
                issues.append(_issue(None, "shortcut", "bank",
                                     f"{shortcut!r} is declared forbidden yet every recipe in the bank uses {artifact!r}"))
    return {"recipe_count": len(recipes), "passed": not issues, "issue_count": len(issues),
            "issues": [issue.as_dict() for issue in issues],
            "stages": list(VALIDATION_STAGES),
            "duplicate_recipe_ids": sorted(key for key, count in seen_ids.items() if count > 1),
            "unique_recipe_hashes": len(seen_hashes)}
