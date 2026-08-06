from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from .schema import ARTIFACTS, GEOMETRY_SHAPES, ILLUMINATION, MEDIA, REGIONS, ROUTES


class OntologyError(ValueError):
    """The ontology file is inconsistent with the recipe schema vocabulary."""


@dataclass(frozen=True)
class Range:
    minimum: float
    maximum: float
    def contains(self, value: float) -> bool: return self.minimum <= float(value) <= self.maximum
    def clip(self, value: float) -> float: return float(min(max(float(value), self.minimum), self.maximum))
    def as_dict(self) -> dict[str, float]: return {"min": self.minimum, "max": self.maximum}


def _range(payload: dict[str, Any], name: str) -> Range:
    if not isinstance(payload, dict) or "min" not in payload or "max" not in payload:
        raise OntologyError(f"range {name!r} must define min and max")
    low, high = float(payload["min"]), float(payload["max"])
    if low > high: raise OntologyError(f"range {name!r} has min > max")
    return Range(low, high)


@dataclass(frozen=True)
class Ontology:
    """Source-only attack ontology: the single visible home of every allowed
    value, safe range and compatibility rule used by M7."""
    version: str
    recipe_schema_version: str
    media: tuple[str, ...]
    geometry_shapes: tuple[str, ...]
    regions: tuple[str, ...]
    artifacts: tuple[str, ...]
    illumination: tuple[str, ...]
    routes: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...]
    limits: dict[str, Any]
    medium_ranges: dict[str, Range]
    geometry_ranges: dict[str, Range]
    capture_ranges: dict[str, Range]
    artifact_strength_ranges: dict[str, Range]
    artifact_parameter_ranges: dict[str, Range]
    medium_artifact_compatibility: dict[str, tuple[str, ...]]
    geometry_region_compatibility: dict[str, tuple[str, ...]]
    aliases: dict[str, dict[str, Any]]
    forbidden_tokens: tuple[str, ...]
    forbidden_token_patterns: tuple[str, ...]
    source_text: str
    sha256: str
    path: Path | None = None

    # --- vocabulary ---------------------------------------------------------
    @property
    def region_order(self) -> dict[str, int]: return {name: index for index, name in enumerate(self.regions)}
    def sort_regions(self, names: list[str]) -> list[str]:
        order = self.region_order
        return sorted(dict.fromkeys(names), key=order.__getitem__)
    def artifact_index(self, name: str) -> int: return self.artifacts.index(name)
    def max_abs_yaw(self) -> float:
        yaw = self.capture_ranges["yaw"]
        return max(abs(yaw.minimum), abs(yaw.maximum))

    # --- compatibility ------------------------------------------------------
    def artifacts_for_medium(self, family: str) -> tuple[str, ...]:
        try: return self.medium_artifact_compatibility[family]
        except KeyError: raise OntologyError(f"medium {family!r} has no compatibility entry") from None
    def regions_for_geometry(self, shape: str) -> tuple[str, ...]:
        try: return self.geometry_region_compatibility[shape]
        except KeyError: raise OntologyError(f"geometry {shape!r} has no compatibility entry") from None
    def strength_range(self, artifact: str) -> Range:
        try: return self.artifact_strength_ranges[artifact]
        except KeyError: raise OntologyError(f"artifact {artifact!r} has no safe strength range") from None
    def parameter_range(self, artifact: str, key: str) -> Range | None:
        return self.artifact_parameter_ranges.get(f"{artifact}.{key}")
    def parameter_keys(self, artifact: str) -> tuple[str, ...]:
        prefix = f"{artifact}."
        return tuple(sorted(key[len(prefix):] for key in self.artifact_parameter_ranges if key.startswith(prefix)))

    # --- alias canonicalization --------------------------------------------
    def canonical_medium(self, value: str) -> str: return self._canonical_scalar("media", value, self.media)
    def canonical_geometry(self, value: str) -> str: return self._canonical_scalar("geometry_shapes", value, self.geometry_shapes)
    def canonical_artifact(self, value: str) -> str: return self._canonical_scalar("artifacts", value, self.artifacts)
    def canonical_illumination(self, value: str) -> str: return self._canonical_scalar("illumination", value, self.illumination)
    def _canonical_scalar(self, group: str, value: str, allowed: tuple[str, ...]) -> str:
        key = str(value).strip()
        if key in allowed: return key
        mapping = self.aliases.get(group, {})
        resolved = mapping.get(key) or mapping.get(key.lower())
        if isinstance(resolved, str) and resolved in allowed: return resolved
        raise OntologyError(f"{group[:-1] if group.endswith('s') else group} {value!r} is not canonical and has no alias")
    def canonical_regions(self, values: list[str]) -> list[str]:
        mapping = self.aliases.get("regions", {})
        expanded: list[str] = []
        for value in values:
            key = str(value).strip()
            if key in self.regions: expanded.append(key); continue
            resolved = mapping.get(key) or mapping.get(key.lower())
            if isinstance(resolved, str): resolved = [resolved]
            if not isinstance(resolved, list) or not resolved or any(name not in self.regions for name in resolved):
                raise OntologyError(f"region {value!r} is not canonical and has no alias")
            expanded.extend(resolved)
        return self.sort_regions(expanded)

    # --- leakage ------------------------------------------------------------
    def leakage_hits(self, text: str) -> list[str]:
        """Forbidden dataset/path/private tokens found in a recipe's text form."""
        lowered = text.lower()
        hits = [token for token in self.forbidden_tokens if token.lower() in lowered]
        hits += [pattern for pattern in self.forbidden_token_patterns if re.search(pattern, text)]
        return sorted(dict.fromkeys(hits))

    def summary(self) -> dict[str, Any]:
        return {"ontology_version": self.version, "ontology_sha256": self.sha256,
                "recipe_schema_version": self.recipe_schema_version, "media": list(self.media),
                "geometry_shapes": list(self.geometry_shapes), "regions": list(self.regions),
                "artifacts": list(self.artifacts), "illumination": list(self.illumination),
                "routes": list(self.routes), "limits": dict(self.limits)}


def load_ontology(path: Path) -> Ontology:
    """Load and self-check the ontology. The SHA-256 is taken over the exact
    file bytes so a frozen bank can prove which ontology produced it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_ontology(text, path=path)


def parse_ontology(text: str, *, path: Path | None = None) -> Ontology:
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict): raise OntologyError("ontology must be a YAML mapping")
    missing = [key for key in ("ontology_version", "recipe_schema_version", "media", "geometry_shapes", "regions",
                               "artifacts", "illumination", "routes", "limits", "medium_ranges", "geometry_ranges",
                               "capture_ranges", "artifact_strength_ranges", "medium_artifact_compatibility",
                               "geometry_region_compatibility", "aliases", "forbidden_tokens") if key not in payload]
    if missing: raise OntologyError(f"ontology is missing keys: {missing}")
    ontology = Ontology(
        version=str(payload["ontology_version"]), recipe_schema_version=str(payload["recipe_schema_version"]),
        media=tuple(payload["media"]), geometry_shapes=tuple(payload["geometry_shapes"]),
        regions=tuple(payload["regions"]), artifacts=tuple(payload["artifacts"]),
        illumination=tuple(payload["illumination"]), routes=tuple(payload["routes"]),
        forbidden_shortcuts=tuple(payload.get("forbidden_shortcuts", ())), limits=dict(payload["limits"]),
        medium_ranges={key: _range(value, key) for key, value in payload["medium_ranges"].items()},
        geometry_ranges={key: _range(value, key) for key, value in payload["geometry_ranges"].items()},
        capture_ranges={key: _range(value, key) for key, value in payload["capture_ranges"].items()},
        artifact_strength_ranges={key: _range(value, key) for key, value in payload["artifact_strength_ranges"].items()},
        artifact_parameter_ranges={key: _range(value, key) for key, value in payload.get("artifact_parameter_ranges", {}).items()},
        medium_artifact_compatibility={key: tuple(value) for key, value in payload["medium_artifact_compatibility"].items()},
        geometry_region_compatibility={key: tuple(value) for key, value in payload["geometry_region_compatibility"].items()},
        aliases={key: dict(value) for key, value in payload["aliases"].items()},
        forbidden_tokens=tuple(payload["forbidden_tokens"]),
        forbidden_token_patterns=tuple(payload.get("forbidden_token_patterns", ())),
        source_text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), path=path)
    _self_check(ontology)
    return ontology


def _self_check(ontology: Ontology) -> None:
    """The ontology may narrow the schema vocabulary but never invent values
    outside it, and it must keep the severity budget satisfiable."""
    for name, values, allowed in (("media", ontology.media, MEDIA), ("geometry_shapes", ontology.geometry_shapes, GEOMETRY_SHAPES),
                                  ("regions", ontology.regions, REGIONS), ("artifacts", ontology.artifacts, ARTIFACTS),
                                  ("illumination", ontology.illumination, ILLUMINATION), ("routes", ontology.routes, ROUTES)):
        if len(set(values)) != len(values): raise OntologyError(f"ontology {name} contains duplicates")
        unknown = [value for value in values if value not in allowed]
        if unknown: raise OntologyError(f"ontology {name} contains values outside the schema vocabulary: {unknown}")
    if tuple(ontology.regions) != REGIONS: raise OntologyError("ontology regions must keep the canonical nine-region order")
    for family in ontology.media:
        unknown = [name for name in ontology.artifacts_for_medium(family) if name not in ontology.artifacts]
        if unknown: raise OntologyError(f"medium {family!r} lists unknown artifacts {unknown}")
    covered = {name for family in ontology.media for name in ontology.artifacts_for_medium(family)}
    uncovered = [name for name in ontology.artifacts if name not in covered]
    if uncovered: raise OntologyError(f"artifacts unreachable from every medium: {uncovered}")
    for shape in ontology.geometry_shapes:
        unknown = [name for name in ontology.regions_for_geometry(shape) if name not in ontology.regions]
        if unknown: raise OntologyError(f"geometry {shape!r} lists unknown regions {unknown}")
    for artifact in ontology.artifacts:
        ontology.strength_range(artifact)
    for key in ontology.artifact_parameter_ranges:
        head = key.split(".", 1)[0]
        if head not in ontology.artifacts: raise OntologyError(f"artifact parameter {key!r} names an unknown artifact")
    limits = ontology.limits
    for key in ("min_artifacts_per_recipe", "max_artifacts_per_recipe", "min_regions_per_recipe",
                "max_regions_per_recipe", "max_total_artifact_strength", "seed_min", "seed_max"):
        if key not in limits: raise OntologyError(f"ontology limits missing {key!r}")
    floor = sum(sorted((ontology.strength_range(name).minimum for name in ontology.artifacts), reverse=True)[:int(limits["max_artifacts_per_recipe"])])
    if floor > float(limits["max_total_artifact_strength"]) + 1e-12:
        raise OntologyError("severity budget is unsatisfiable: minimum safe strengths already exceed max_total_artifact_strength")
