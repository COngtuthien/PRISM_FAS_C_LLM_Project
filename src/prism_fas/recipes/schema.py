from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RECIPE_SCHEMA_VERSION = "1.1"
RECIPE_ID_PATTERN = r"^R-[0-9]{6}$"
SEED_MIN, SEED_MAX = 0, 2_147_483_647
# Structural enums. The ontology narrows these further (ranges, compatibility);
# the schema only refuses values that are not part of the vocabulary at all.
MEDIA = ("paper-like", "display-like", "plastic-like", "fabric-like", "reflective-film-like")
GEOMETRY_SHAPES = ("flat", "curved", "partial-curved", "flexible", "rigid", "boundary-only")
REGIONS = ("left_eye", "right_eye", "nose", "mouth", "forehead", "left_cheek", "right_cheek", "face_boundary", "context")
ARTIFACTS = ("halftone", "pixel_grid", "moire", "specular_reflection", "texture_smoothing", "color_shift",
             "boundary_inconsistency", "blur")
ILLUMINATION = ("front", "left", "right", "top", "bottom", "mixed")
ROUTES = ("physics", "gpat")
FORBIDDEN_SHORTCUTS = ("always_moire", "always_halftone")
REGION_ORDER = {name: index for index, name in enumerate(REGIONS)}


class RecipeSchemaError(ValueError):
    """A recipe payload does not satisfy the strict v1.1 schema."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Medium(_Strict):
    family: str
    transparency: float = Field(ge=0.0, le=1.0)
    roughness: float = Field(ge=0.0, le=1.0)

    @field_validator("family")
    @classmethod
    def _family(cls, value: str) -> str:
        if value not in MEDIA: raise ValueError(f"unknown medium family {value!r}; allowed {list(MEDIA)}")
        return value


class Geometry(_Strict):
    shape: str
    rigidity: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(gt=0.0, le=1.0)

    @field_validator("shape")
    @classmethod
    def _shape(cls, value: str) -> str:
        if value not in GEOMETRY_SHAPES: raise ValueError(f"unknown geometry shape {value!r}; allowed {list(GEOMETRY_SHAPES)}")
        return value


class ArtifactSpec(_Strict):
    """One artifact request. `parameters` is an optional strict override map:
    only documented, ontology-bounded operator keys are accepted."""
    name: str
    strength: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if value not in ARTIFACTS: raise ValueError(f"unknown artifact {value!r}; allowed {list(ARTIFACTS)}")
        return value

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, value: dict[str, Any]) -> dict[str, float]:
        for key, item in value.items():
            if not isinstance(key, str) or not key: raise ValueError("artifact parameter keys must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"artifact parameter {key!r} must be a real number")
            if not (float(item) == float(item)) or abs(float(item)) == float("inf"):
                raise ValueError(f"artifact parameter {key!r} must be finite")
        return {key: float(item) for key, item in value.items()}


class Capture(_Strict):
    yaw: float
    illumination: str
    compression_q: int
    scale: float
    motion: float = Field(ge=0.0, le=1.0)
    defocus: float = Field(ge=0.0, le=1.0)

    @field_validator("illumination")
    @classmethod
    def _illumination(cls, value: str) -> str:
        if value not in ILLUMINATION: raise ValueError(f"unknown illumination {value!r}; allowed {list(ILLUMINATION)}")
        return value

    @field_validator("compression_q")
    @classmethod
    def _compression(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int): raise ValueError("compression_q must be an integer")
        return value


class RecipeV11(_Strict):
    """Strict source-only attack recipe, schema v1.1.

    Range/compatibility rules that depend on the ontology (safe strength bands,
    medium/artifact and geometry/region compatibility, severity budget) are
    enforced by `prism_fas.recipes.validate`, not here: this model owns only the
    structural contract that holds for every ontology version.
    """
    recipe_id: str = Field(pattern=RECIPE_ID_PATTERN)
    medium: Medium
    geometry: Geometry
    regions: list[str] = Field(min_length=1, max_length=3)
    artifacts: list[ArtifactSpec] = Field(min_length=1, max_length=3)
    capture: Capture
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    generator_route: list[str] = Field(min_length=1, max_length=2)
    seed: int = Field(ge=SEED_MIN, le=SEED_MAX)
    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, value: str) -> str:
        if value != RECIPE_SCHEMA_VERSION: raise ValueError(f"schema_version must be {RECIPE_SCHEMA_VERSION!r}, got {value!r}")
        return value

    @field_validator("seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int): raise ValueError("seed must be an integer")
        return value

    @field_validator("regions")
    @classmethod
    def _regions(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in REGION_ORDER]
        if unknown: raise ValueError(f"unknown regions {unknown}; allowed {list(REGIONS)}")
        if len(set(value)) != len(value): raise ValueError(f"duplicate regions in {value}")
        if value != sorted(value, key=REGION_ORDER.__getitem__): raise ValueError(f"regions must use canonical order: {sorted(value, key=REGION_ORDER.__getitem__)}")
        return value

    @field_validator("forbidden_shortcuts")
    @classmethod
    def _shortcuts(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in FORBIDDEN_SHORTCUTS]
        if unknown: raise ValueError(f"unknown forbidden_shortcuts {unknown}; allowed {list(FORBIDDEN_SHORTCUTS)}")
        if len(set(value)) != len(value): raise ValueError(f"duplicate forbidden_shortcuts in {value}")
        return value

    @field_validator("generator_route")
    @classmethod
    def _routes(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in ROUTES]
        if unknown: raise ValueError(f"unknown generator_route {unknown}; allowed {list(ROUTES)}")
        if len(set(value)) != len(value): raise ValueError(f"duplicate generator_route entries in {value}")
        return value

    @model_validator(mode="after")
    def _artifacts_unique(self) -> "RecipeV11":
        names = [artifact.name for artifact in self.artifacts]
        if len(set(names)) != len(names): raise ValueError(f"duplicate artifact names in {names}")
        return self

    def artifact(self, name: str) -> ArtifactSpec | None:
        for spec in self.artifacts:
            if spec.name == name: return spec
        return None

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def parse_recipe(payload: dict[str, Any]) -> RecipeV11:
    """Parse one recipe payload, raising `RecipeSchemaError` on any violation."""
    if not isinstance(payload, dict): raise RecipeSchemaError("recipe payload must be a JSON object")
    try:
        return RecipeV11.model_validate(payload)
    except Exception as exc:
        raise RecipeSchemaError(str(exc)) from exc
