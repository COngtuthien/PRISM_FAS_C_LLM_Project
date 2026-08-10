"""Generic, ontology-level coverage quotas for a batch generation request.

A quota here is an experimental control over the FROZEN ONTOLOGY and nothing
else. It says "across this batch, every medium family must appear at least four
times"; it can never say "prefer the attacks the target set contains", because it
has no vocabulary for that. Every category name it can mention is a value the
ontology already defines, and `QuotaSpec.validate_against` refuses a spec naming
anything else.

That restriction is what keeps the quota source-independent. It carries no
frequency, deficit or error signal from any corpus, benchmark or previous
evaluation - only the batch size and the ontology's own vocabulary.

Two kinds of bound are distinguished on purpose:

* required  - `require_all`, `min_per_category`, `max_per_category`. A miss is a
  quota failure.
* preferred - `preferred_min_per_category`. A miss is recorded and reported, but
  it is not a failure, because physical compatibility outranks a quota: it is
  better to under-fill a category than to emit a recipe a real medium could not
  produce.

The spec is identity-bearing. Changing any bound changes `quota_identity`, which
enters the generation request identity, so a batch generated under different
quotas can never be mistaken for one generated under these.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11

QUOTA_SCHEMA_VERSION = "c2b-coverage-quota-v1"

#: The five audited axes. `multi` marks an axis a single recipe can occupy in
#: more than one category at once, where a count means "recipes carrying it".
QUOTA_AXES: tuple[tuple[str, bool], ...] = (
    ("media", False),
    ("geometry", False),
    ("illumination", False),
    ("artifacts", True),
    ("regions", True),
)


class QuotaError(ValueError):
    """A quota spec is malformed or names something the ontology does not."""


def axis_values(recipe: RecipeV11) -> dict[str, list[str]]:
    """The categories one recipe occupies on each quota axis."""
    return {
        "media": [recipe.medium.family],
        "geometry": [recipe.geometry.shape],
        "illumination": [recipe.capture.illumination],
        "artifacts": sorted({spec.name for spec in recipe.artifacts}),
        "regions": sorted(set(recipe.regions)),
    }


def _plural(count: int) -> str:
    return "recipe" if count == 1 else "recipes"


def axis_vocabulary(ontology: Ontology) -> dict[str, tuple[str, ...]]:
    return {
        "media": tuple(ontology.media),
        "geometry": tuple(ontology.geometry_shapes),
        "illumination": tuple(ontology.illumination),
        "artifacts": tuple(ontology.artifacts),
        "regions": tuple(ontology.regions),
    }


@dataclass(frozen=True)
class AxisQuota:
    """Bounds for one axis, in recipe counts across the batch."""

    axis: str
    require_all: bool = True
    min_per_category: int | None = None
    preferred_min_per_category: int | None = None
    max_per_category: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuotaSpec:
    """The whole batch-level control. Immutable and hashable."""

    batch_size: int
    axes: tuple[AxisQuota, ...]
    schema_version: str = QUOTA_SCHEMA_VERSION
    label: str = ""
    diversity_rules: tuple[str, ...] = field(default_factory=tuple)

    def axis(self, name: str) -> AxisQuota | None:
        for quota in self.axes:
            if quota.axis == name:
                return quota
        return None

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_size": self.batch_size,
            "label": self.label,
            "axes": [quota.as_dict() for quota in
                     sorted(self.axes, key=lambda item: item.axis)],
            "diversity_rules": list(self.diversity_rules),
        }

    @property
    def quota_identity(self) -> str:
        text = json.dumps(self.identity_material(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_material(), "quota_identity": self.quota_identity}

    # --- self-checks --------------------------------------------------------
    def validate_against(self, ontology: Ontology) -> None:
        """Fail closed on an axis the ontology does not have, or on a bound that
        cannot be satisfied by a batch of this size."""
        vocabulary = axis_vocabulary(ontology)
        seen: set[str] = set()
        for quota in self.axes:
            if quota.axis not in vocabulary:
                raise QuotaError(f"unknown quota axis {quota.axis!r}; "
                                 f"allowed {sorted(vocabulary)}")
            if quota.axis in seen:
                raise QuotaError(f"duplicate quota for axis {quota.axis!r}")
            seen.add(quota.axis)
            size = len(vocabulary[quota.axis])
            multi = dict(QUOTA_AXES)[quota.axis]
            for name in ("min_per_category", "preferred_min_per_category", "max_per_category"):
                value = getattr(quota, name)
                if value is not None and (not isinstance(value, int) or value < 0):
                    raise QuotaError(f"{quota.axis}.{name} must be a non-negative integer")
            if (quota.min_per_category is not None and quota.max_per_category is not None
                    and quota.min_per_category > quota.max_per_category):
                raise QuotaError(f"{quota.axis}: min_per_category exceeds max_per_category")
            if multi:
                continue
            # A single-valued axis partitions the batch, so its bounds must
            # admit a partition of exactly `batch_size` recipes.
            if quota.min_per_category is not None and quota.min_per_category * size > self.batch_size:
                raise QuotaError(
                    f"{quota.axis}: min {quota.min_per_category} x {size} categories exceeds "
                    f"the batch size {self.batch_size}; the quota is unsatisfiable")
            if quota.max_per_category is not None and quota.max_per_category * size < self.batch_size:
                raise QuotaError(
                    f"{quota.axis}: max {quota.max_per_category} x {size} categories cannot "
                    f"cover the batch size {self.batch_size}; the quota is unsatisfiable")

    # --- prompt rendering ---------------------------------------------------
    def prompt_block(self, ontology: Ontology) -> dict[str, str]:
        """The coverage quotas as the request template's `coverage_quotas` map.

        Only ontology vocabulary and counts appear. The text is generated from
        the spec, so the prompt the model sees and the audit that judges it can
        never drift apart.
        """
        vocabulary = axis_vocabulary(ontology)
        block: dict[str, str] = {}
        for quota in sorted(self.axes, key=lambda item: item.axis):
            names = vocabulary[quota.axis]
            multi = dict(QUOTA_AXES)[quota.axis]
            # A single-valued axis partitions the batch ("N recipes use it"); a
            # multi-valued one is a membership count ("it appears in N recipes").
            parts: list[str] = []
            if quota.require_all:
                parts.append(f"every one of the {len(names)} values must appear "
                             f"({', '.join(names)})")
            if quota.min_per_category:
                parts.append(f"each value must appear in at least {quota.min_per_category} "
                             f"{_plural(quota.min_per_category)}" if multi
                             else f"at least {quota.min_per_category} "
                                  f"{_plural(quota.min_per_category)} per value")
            if quota.preferred_min_per_category:
                parts.append(
                    f"preferably each value appears in at least "
                    f"{quota.preferred_min_per_category} "
                    f"{_plural(quota.preferred_min_per_category)}, where physical "
                    "compatibility permits")
            if quota.max_per_category is not None:
                parts.append(f"no value may appear in more than {quota.max_per_category} "
                             f"{_plural(quota.max_per_category)}")
            if quota.note:
                parts.append(quota.note)
            block[quota.axis] = "; ".join(parts)
        if self.diversity_rules:
            block["diversity"] = "; ".join(self.diversity_rules)
        block["precedence"] = ("physical compatibility outranks every quota above - if a value "
                               "cannot be used plausibly, under-fill it rather than emit an "
                               "implausible recipe, and never invent a value outside the "
                               "vocabulary")
        return block


# ------------------------------------------------------------------ compliance
def evaluate(spec: QuotaSpec, recipes: Sequence[RecipeV11],
             ontology: Ontology) -> dict[str, Any]:
    """Per-category counts and pass/fail against the spec. Deterministic.

    Nothing here modifies a recipe. A quota result is a measurement of what the
    provider produced, never a reason to move a recipe between categories.
    """
    vocabulary = axis_vocabulary(ontology)
    total = len(recipes)
    axes: dict[str, Any] = {}
    required_failures: list[dict[str, Any]] = []
    preferred_misses: list[dict[str, Any]] = []

    for quota in sorted(spec.axes, key=lambda item: item.axis):
        names = vocabulary[quota.axis]
        counts = {name: 0 for name in names}
        for recipe in recipes:
            for value in axis_values(recipe)[quota.axis]:
                counts[value] += 1
        assignments = sum(counts.values())

        categories: dict[str, Any] = {}
        for name in names:
            count = counts[name]
            failures: list[str] = []
            misses: list[str] = []
            if quota.require_all and count == 0:
                failures.append("required to appear at least once")
            if quota.min_per_category is not None and count < quota.min_per_category:
                failures.append(f"below the required minimum {quota.min_per_category}")
            if quota.max_per_category is not None and count > quota.max_per_category:
                failures.append(f"above the maximum {quota.max_per_category}")
            if (quota.preferred_min_per_category is not None
                    and count < quota.preferred_min_per_category
                    and not failures):
                misses.append(f"below the preferred minimum {quota.preferred_min_per_category}")
            categories[name] = {
                "count": count,
                "percent_of_recipes": round(100.0 * count / total, 4) if total else 0.0,
                "percent_of_axis": round(100.0 * count / assignments, 4) if assignments else 0.0,
                "quota_minimum": quota.min_per_category,
                "quota_preferred_minimum": quota.preferred_min_per_category,
                "quota_maximum": quota.max_per_category,
                "required_pass": not failures,
                "preferred_pass": not misses,
                "failures": failures,
                "preferred_misses": misses,
            }
            for reason in failures:
                required_failures.append({"axis": quota.axis, "category": name,
                                          "count": count, "reason": reason})
            for reason in misses:
                preferred_misses.append({"axis": quota.axis, "category": name,
                                         "count": count, "reason": reason})

        present = [name for name in names if counts[name] > 0]
        top = max(categories.items(), key=lambda item: item[1]["count"]) if categories else None
        axes[quota.axis] = {
            "quota": quota.as_dict(),
            "category_count": len(names),
            "categories_present": len(present),
            "categories_missing": [name for name in names if counts[name] == 0],
            "coverage_fraction": round(len(present) / len(names), 6) if names else 0.0,
            "assignment_total": assignments,
            "max_share_percent": top[1]["percent_of_axis"] if top else 0.0,
            "max_share_category": top[0] if top else None,
            "required_pass": all(entry["required_pass"] for entry in categories.values()),
            "preferred_pass": all(entry["preferred_pass"] for entry in categories.values()),
            "categories": categories,
        }

    return {
        "schema_version": QUOTA_SCHEMA_VERSION,
        "quota_identity": spec.quota_identity,
        "batch_size_expected": spec.batch_size,
        "recipes_evaluated": total,
        "axes": axes,
        "required_pass": not required_failures,
        "preferred_pass": not preferred_misses,
        "required_failures": required_failures,
        "preferred_misses": preferred_misses,
        "policy": "compatibility outranks quota; a preferred miss is reported, never repaired, "
                  "and no recipe was moved between categories after generation",
    }


def classify_recipes(spec: QuotaSpec, recipes: Sequence[RecipeV11],
                     ontology: Ontology) -> dict[str, Any]:
    """Split accepted recipes into quota-compliant and quota-miss.

    A recipe is `VALID_BUT_QUOTA_MISS` when it is the sole occupant of a category
    that failed a required bound, or when it sits in an over-filled category.
    The label describes the batch's shape; it never alters the recipe.
    """
    result = evaluate(spec, recipes, ontology)
    offending: dict[str, set[str]] = {}
    for failure in result["required_failures"]:
        offending.setdefault(failure["axis"], set()).add(failure["category"])

    rows: list[dict[str, Any]] = []
    for index, recipe in enumerate(recipes):
        reasons: list[str] = []
        values = axis_values(recipe)
        for axis, categories in offending.items():
            for value in values[axis]:
                if value in categories:
                    reasons.append(f"{axis}={value} is in a category that missed a required bound")
        rows.append({
            "index": index,
            "recipe_id": recipe.recipe_id,
            "classification": "VALID_BUT_QUOTA_MISS" if reasons else "VALID_AND_QUOTA_COMPLIANT",
            "reasons": sorted(set(reasons)),
        })
    return {"rows": rows,
            "valid_and_quota_compliant": sum(1 for row in rows
                                             if row["classification"] == "VALID_AND_QUOTA_COMPLIANT"),
            "valid_but_quota_miss": sum(1 for row in rows
                                        if row["classification"] == "VALID_BUT_QUOTA_MISS")}


# ---------------------------------------------------------------------- loading
_ALLOWED_AXIS_KEYS = {"axis", "require_all", "min_per_category", "preferred_min_per_category",
                      "max_per_category", "note"}
_ALLOWED_SPEC_KEYS = {"schema_version", "batch_size", "label", "axes", "diversity_rules"}


def parse_quota_spec(payload: Any) -> QuotaSpec:
    """Strict parse. An unknown key is rejected rather than ignored, so a typo
    cannot silently drop a bound the batch was supposed to carry."""
    if not isinstance(payload, dict):
        raise QuotaError("quota spec must be a mapping")
    unknown = sorted(set(payload) - _ALLOWED_SPEC_KEYS)
    if unknown:
        raise QuotaError(f"unknown quota spec keys {unknown}")
    if "batch_size" not in payload or "axes" not in payload:
        raise QuotaError("quota spec requires batch_size and axes")
    axes: list[AxisQuota] = []
    for entry in payload["axes"]:
        if not isinstance(entry, dict):
            raise QuotaError("each axis quota must be a mapping")
        extra = sorted(set(entry) - _ALLOWED_AXIS_KEYS)
        if extra:
            raise QuotaError(f"unknown axis quota keys {extra}")
        if "axis" not in entry:
            raise QuotaError("each axis quota requires an 'axis' name")
        axes.append(AxisQuota(**entry))
    spec = QuotaSpec(
        batch_size=int(payload["batch_size"]),
        axes=tuple(axes),
        schema_version=str(payload.get("schema_version", QUOTA_SCHEMA_VERSION)),
        label=str(payload.get("label", "")),
        diversity_rules=tuple(payload.get("diversity_rules", ())),
    )
    if spec.schema_version != QUOTA_SCHEMA_VERSION:
        raise QuotaError(f"unsupported quota schema_version {spec.schema_version!r}")
    return spec


def load_quota_spec(path: Any) -> QuotaSpec:
    import yaml
    from pathlib import Path
    return parse_quota_spec(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


__all__ = ["QUOTA_SCHEMA_VERSION", "QUOTA_AXES", "AxisQuota", "QuotaSpec", "QuotaError",
           "axis_values", "axis_vocabulary", "evaluate", "classify_recipes",
           "parse_quota_spec", "load_quota_spec"]
