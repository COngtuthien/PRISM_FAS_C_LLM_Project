"""Deterministic audits over a completed LLM recipe pilot.

Every function here is pure: given the same recipes and the same attempt records
it returns byte-identical JSON. Nothing in this module calls a provider, reads a
credential, touches the network or looks at any dataset. The audits describe what
the frozen prompt/schema/ontology contract actually produced; they do not rate a
recipe against any target, corpus or attack taxonomy.

Three families of audit live here:

* coverage      - which ontology categories the pilot reached, on five axes
* co-occurrence - artifact x medium, artifact x geometry, artifact x region
* statistics    - validity, retry, duplicate, error and latency rates

The flags raised by `flag_pilot_issues` are objective and source-independent
(share of a category, exact repeats, compatibility rejections). They are
deliberately not quality judgements: a recipe is never scored for usefulness.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Sequence

from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11

AUDIT_SCHEMA_VERSION = "c2-pilot-audit-v1"

#: The five coverage axes named by the C2 instruction. `multi` marks an axis a
#: single recipe can occupy in more than one category at once.
COVERAGE_AXES: tuple[tuple[str, bool], ...] = (
    ("artifacts", True),
    ("regions", True),
    ("media", False),
    ("geometry", False),
    ("illumination", False),
)

#: Objective flag thresholds. Stated as constants so the audit is reproducible
#: and so a reader can see exactly what "collapse" and "low coverage" mean here.
MODE_COLLAPSE_SHARE = 0.60          # one category holds >= 60% of the axis mass
LOW_COVERAGE_FRACTION = 0.50        # fewer than half an axis's categories appear
REPEATED_PATTERN_MIN_COUNT = 3      # the same structural pattern >= 3 times
SEVERITY_BUDGET_MARGIN = 0.02       # "at the budget ceiling" band, absolute


def axis_categories(ontology: Ontology) -> dict[str, tuple[str, ...]]:
    """The full category list for each coverage axis, from the frozen ontology."""
    return {
        "artifacts": tuple(ontology.artifacts),
        "regions": tuple(ontology.regions),
        "media": tuple(ontology.media),
        "geometry": tuple(ontology.geometry_shapes),
        "illumination": tuple(ontology.illumination),
    }


def _axis_values(recipe: RecipeV11) -> dict[str, list[str]]:
    """The categories one recipe occupies on each axis."""
    return {
        "artifacts": [spec.name for spec in recipe.artifacts],
        "regions": list(recipe.regions),
        "media": [recipe.medium.family],
        "geometry": [recipe.geometry.shape],
        "illumination": [recipe.capture.illumination],
    }


# --------------------------------------------------------------------- coverage
def coverage_audit(recipes: Sequence[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    """Per-axis category counts, percentages and missing categories.

    Percentages are reported two ways, because they answer different questions:
    `recipe_percent` is the share of recipes that carry the category at all, and
    `assignment_percent` is the category's share of that axis's total mass.
    """
    categories = axis_categories(ontology)
    total = len(recipes)
    axes: dict[str, Any] = {}

    for axis, multi in COVERAGE_AXES:
        allowed = categories[axis]
        recipe_hits: Counter[str] = Counter()
        assignments: Counter[str] = Counter()
        for recipe in recipes:
            values = _axis_values(recipe)[axis]
            assignments.update(values)
            recipe_hits.update(sorted(set(values)))
        assignment_total = sum(assignments.values())
        present = [name for name in allowed if recipe_hits[name] > 0]
        missing = [name for name in allowed if recipe_hits[name] == 0]
        axes[axis] = {
            "multi_valued": multi,
            "category_count": len(allowed),
            "categories_present": len(present),
            "categories_missing": len(missing),
            "coverage_fraction": round(len(present) / len(allowed), 6) if allowed else 0.0,
            "missing_categories": missing,
            "counts": {name: int(recipe_hits[name]) for name in allowed},
            "assignment_counts": {name: int(assignments[name]) for name in allowed},
            "recipe_percent": {name: round(100.0 * recipe_hits[name] / total, 4) if total else 0.0
                               for name in allowed},
            "assignment_percent": {
                name: round(100.0 * assignments[name] / assignment_total, 4) if assignment_total else 0.0
                for name in allowed},
            "assignment_total": assignment_total,
        }

    artifacts_per = [len(recipe.artifacts) for recipe in recipes]
    regions_per = [len(recipe.regions) for recipe in recipes]
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "recipe_count": total,
        "ontology_identity": ontology.sha256,
        "axes": axes,
        "artifacts_per_recipe": _distribution(artifacts_per),
        "regions_per_recipe": _distribution(regions_per),
    }


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    counts = Counter(values)
    total = len(values)
    return {
        "n": total,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(sum(values) / total, 6) if total else None,
        "histogram": {str(key): int(counts[key]) for key in sorted(counts)},
        "percent": {str(key): round(100.0 * counts[key] / total, 4) for key in sorted(counts)} if total else {},
    }


# ---------------------------------------------------------------- co-occurrence
def cooccurrence_audit(recipes: Sequence[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    """artifact x medium, artifact x geometry and artifact x region tables.

    Cells count (recipe, artifact, other-axis-value) incidences, so a recipe with
    two artifacts over two regions contributes four incidences to the region
    table. Compatibility-impossible cells are marked, which is what makes a zero
    interpretable: an impossible zero is not a coverage gap.
    """
    artifacts = tuple(ontology.artifacts)
    tables: dict[str, Any] = {}

    def build(name: str, columns: tuple[str, ...], allowed: dict[str, tuple[str, ...]],
              column_values) -> None:
        grid = {row: {column: 0 for column in columns} for row in artifacts}
        for recipe in recipes:
            for spec in recipe.artifacts:
                for column in column_values(recipe):
                    grid[spec.name][column] += 1
        # A cell is reachable only if some ontology entry permits the pairing.
        reachable = {row: {column: (row in allowed.get(column, ())) for column in columns}
                     for row in artifacts} if name == "artifact_x_medium" else None
        table: dict[str, Any] = {
            "rows": list(artifacts),
            "columns": list(columns),
            "cells": grid,
            "row_totals": {row: sum(grid[row].values()) for row in artifacts},
            "column_totals": {column: sum(grid[row][column] for row in artifacts) for column in columns},
            "occupied_cells": sum(1 for row in artifacts for column in columns if grid[row][column] > 0),
            "total_cells": len(artifacts) * len(columns),
        }
        if reachable is not None:
            table["compatible_cells"] = reachable
            table["compatible_cell_count"] = sum(
                1 for row in artifacts for column in columns if reachable[row][column])
            table["occupied_compatible_cells"] = sum(
                1 for row in artifacts for column in columns
                if reachable[row][column] and grid[row][column] > 0)
        tables[name] = table

    build("artifact_x_medium", tuple(ontology.media), ontology.medium_artifact_compatibility,
          lambda recipe: [recipe.medium.family])
    build("artifact_x_geometry", tuple(ontology.geometry_shapes), {},
          lambda recipe: [recipe.geometry.shape])
    build("artifact_x_region", tuple(ontology.regions), {},
          lambda recipe: list(recipe.regions))

    return {"audit_schema_version": AUDIT_SCHEMA_VERSION,
            "recipe_count": len(recipes),
            "ontology_identity": ontology.sha256,
            "tables": tables}


# -------------------------------------------------------------------- duplicates
def structural_pattern(recipe: RecipeV11) -> str:
    """The categorical shape of a recipe, with all continuous values dropped.

    Two recipes sharing a pattern are not duplicates - their severities and
    capture parameters still differ - but a pattern repeated many times is an
    objective diversity signal over the categorical vocabulary.
    """
    material = {
        "medium": recipe.medium.family,
        "geometry": recipe.geometry.shape,
        "illumination": recipe.capture.illumination,
        "regions": list(recipe.regions),
        "artifacts": sorted(spec.name for spec in recipe.artifacts),
        "routes": sorted(recipe.generator_route),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def duplicate_audit(content_identities: Sequence[str],
                    recipes: Sequence[RecipeV11] | None = None) -> dict[str, Any]:
    """Exact-content duplicates plus repeated categorical patterns.

    `content_identities` are the canonical content hashes the pipeline computed
    (positional `recipe_id` excluded), so this audit uses exactly the identity
    the accept/reject decision used.
    """
    counts = Counter(content_identities)
    duplicates = {key: int(value) for key, value in sorted(counts.items()) if value > 1}
    patterns = Counter(structural_pattern(recipe) for recipe in (recipes or ()))
    repeated = {key: int(value) for key, value in sorted(patterns.items()) if value > 1}
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "accepted_recipe_count": len(content_identities),
        "unique_content_identities": len(counts),
        "exact_duplicate_groups": len(duplicates),
        "exact_duplicate_identities": duplicates,
        "exact_duplicate_rate": round(
            (len(content_identities) - len(counts)) / len(content_identities), 6)
        if content_identities else 0.0,
        "distinct_structural_patterns": len(patterns),
        "repeated_structural_patterns": repeated,
        "max_structural_pattern_count": max(patterns.values()) if patterns else 0,
    }


# -------------------------------------------------------------------- statistics
def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolated percentile on the sorted sample. Deterministic."""
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    if len(numbers) == 1:
        return round(numbers[0], 6)
    position = fraction * (len(numbers) - 1)
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    weight = position - lower
    return round(numbers[lower] * (1.0 - weight) + numbers[upper] * weight, 6)


def latency_stats(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None,
                "total": 0.0}
    return {"n": len(numbers),
            "mean": round(sum(numbers) / len(numbers), 6),
            "median": percentile(numbers, 0.5),
            "p95": percentile(numbers, 0.95),
            "min": round(min(numbers), 6),
            "max": round(max(numbers), 6),
            "total": round(sum(numbers), 6)}


def token_totals(usages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum whatever token fields the surface actually reported.

    The Interactions API does not guarantee a fixed usage shape, so nothing is
    invented: a field absent from every response stays absent from the total, and
    `reported` records how many attempts carried usage at all.
    """
    totals: Counter[str] = Counter()
    reported = 0
    for usage in usages:
        if not usage:
            continue
        reported += 1
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
    return {"attempts_with_usage": reported,
            "totals": {key: totals[key] for key in sorted(totals)},
            "available": bool(totals)}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def pilot_statistics(slots: Sequence[dict[str, Any]],
                     attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-slot and per-attempt records into the C2 audit numbers.

    A slot is one scientific request slot; a failed attempt never creates a new
    slot, so `slot_count` is fixed and retries live inside their slot.
    """
    slot_count = len(slots)
    successful = [slot for slot in slots if slot["final_status"] == "accepted"]
    exhausted = [slot for slot in slots if slot["final_status"] != "accepted"]
    first_attempt_valid = [slot for slot in successful if slot["accepted_on_attempt"] == 1]
    retried = [slot for slot in slots if slot["semantic_attempts"] > 1]

    outcome_counts = Counter(attempt.get("candidate_outcome") for attempt in attempts
                             if attempt.get("candidate_outcome"))
    stage_counts: Counter[str] = Counter()
    # The inherited validator reports a parse failure and a range failure under
    # the same "schema" stage, separated only by the field: `recipe` means the
    # payload did not parse, anything else names the field whose value is out of
    # band. Conflating them would hide which one the model actually got wrong.
    schema_parse_failures = 0
    range_failures = 0
    for attempt in attempts:
        for issue in attempt.get("issues", ()):
            stage = str(issue.get("stage", "unknown"))
            field = str(issue.get("field", ""))
            stage_counts[stage] += 1
            if stage == "schema":
                if field == "recipe":
                    schema_parse_failures += 1
                else:
                    range_failures += 1
            elif stage == "strength_range":
                range_failures += 1

    error_counts = Counter(attempt["error_class"] for attempt in attempts
                           if attempt.get("error_class"))
    semantic_invalid = sum(1 for attempt in attempts
                           if attempt.get("response_received") and not attempt.get("all_accepted"))

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "slots": {
            "slot_count": slot_count,
            "successful_slots": len(successful),
            "failed_or_exhausted_slots": len(exhausted),
            "exhausted_slot_ids": [slot["slot_id"] for slot in exhausted],
            "slots_that_retried": len(retried),
        },
        "calls": {
            "total_provider_calls": len(attempts),
            "calls_per_slot_mean": round(len(attempts) / slot_count, 6) if slot_count else 0.0,
            "responses_received": sum(1 for a in attempts if a.get("response_received")),
            "transport_or_api_errors": sum(1 for a in attempts if a.get("error_class")),
        },
        "rates": {
            "first_attempt_valid_rate": _rate(len(first_attempt_valid), slot_count),
            "eventual_valid_rate": _rate(len(successful), slot_count),
            "invalid_rate": _rate(semantic_invalid, len(attempts)),
            "retry_rate": _rate(len(retried), slot_count),
            "retry_exhaustion_rate": _rate(len(exhausted), slot_count),
            "duplicate_rate": _rate(int(outcome_counts.get("rejected_duplicate", 0)), len(attempts)),
        },
        "counts": {
            "first_attempt_valid_slots": len(first_attempt_valid),
            "semantic_invalid_responses": semantic_invalid,
            "candidate_outcomes": {key: int(value) for key, value in sorted(outcome_counts.items())},
            "rejected_candidates_by_outcome": {
                key: int(value) for key, value in sorted(outcome_counts.items())
                if key != "accepted"},
            "schema_violations": schema_parse_failures,
            "ontology_violations": int(stage_counts.get("canonical", 0)),
            "envelope_violations": int(outcome_counts.get("rejected_envelope", 0)),
            "system_owned_field_violations": int(outcome_counts.get("rejected_system_owned_field", 0)),
            "duplicate_violations": int(outcome_counts.get("rejected_duplicate", 0)),
            "json_parse_failures": sum(1 for a in attempts
                                       if any(i.get("stage") == "json_parsing"
                                              for i in a.get("issues", ()))),
            "validator_stage_hits": {key: int(stage_counts[key]) for key in sorted(stage_counts)},
            "range_violations": range_failures,
            "compatibility_violations": int(stage_counts.get("medium_artifact", 0)
                                            + stage_counts.get("geometry_region", 0)),
            "severity_budget_violations": int(stage_counts.get("severity", 0)),
            "compiler_failures": sum(1 for slot in slots if slot.get("compiler_status") == "failed"),
            "transport_error_classes": {key: int(value) for key, value in sorted(error_counts.items())},
            "transient_rate_limit_429": int(error_counts.get("rate_limit", 0)),
            "quota_exhausted": int(error_counts.get("quota_exhausted", 0)),
        },
        "latency_seconds": latency_stats(a.get("latency_seconds") for a in attempts),
        "token_usage": token_totals(a.get("usage") or {} for a in attempts),
    }


# ------------------------------------------------------------------------ flags
def flag_pilot_issues(coverage: dict[str, Any], cooccurrence: dict[str, Any],
                      duplicates: dict[str, Any], statistics: dict[str, Any],
                      recipes: Sequence[RecipeV11], ontology: Ontology) -> list[dict[str, Any]]:
    """Objective pilot flags. Never a judgement about a recipe's usefulness."""
    flags: list[dict[str, Any]] = []

    for axis, _multi in COVERAGE_AXES:
        entry = coverage["axes"][axis]
        if entry["coverage_fraction"] < LOW_COVERAGE_FRACTION:
            flags.append({
                "flag": "LOW_COVERAGE", "axis": axis,
                "detail": f"{entry['categories_present']}/{entry['category_count']} categories present; "
                          f"missing {entry['missing_categories']}",
                "threshold": f"coverage_fraction < {LOW_COVERAGE_FRACTION}"})
        # Only the largest category can cross the share threshold, so the axis
        # needs exactly one comparison.
        shares = entry["assignment_percent"]
        if shares:
            name, percent = max(sorted(shares.items()), key=lambda item: item[1])
            if percent >= MODE_COLLAPSE_SHARE * 100.0:
                flags.append({
                    "flag": "MODE_COLLAPSE", "axis": axis, "category": name,
                    "detail": f"{name!r} holds {percent}% of the {axis} assignments",
                    "threshold": f"assignment share >= {MODE_COLLAPSE_SHARE * 100.0}%"})

    for pattern, count in sorted(duplicates["repeated_structural_patterns"].items(),
                                 key=lambda item: (-item[1], item[0])):
        if count >= REPEATED_PATTERN_MIN_COUNT:
            flags.append({"flag": "REPEATED_PATTERN", "count": count, "pattern": json.loads(pattern),
                          "threshold": f"identical categorical pattern >= {REPEATED_PATTERN_MIN_COUNT} times"})

    if statistics["counts"]["compatibility_violations"] > 0:
        flags.append({
            "flag": "COMPATIBILITY_RETRY_REQUIRED",
            "detail": f"{statistics['counts']['compatibility_violations']} medium/artifact or "
                      "geometry/region compatibility rejections forced a retry",
            "threshold": "compatibility_violations > 0"})

    budget = float(ontology.limits["max_total_artifact_strength"])
    for recipe in recipes:
        total = sum(float(spec.strength) for spec in recipe.artifacts)
        if total >= budget - SEVERITY_BUDGET_MARGIN:
            flags.append({
                "flag": "ODD_BUT_VALID", "recipe_id": recipe.recipe_id,
                "detail": f"total artifact strength {round(total, 6)} sits within "
                          f"{SEVERITY_BUDGET_MARGIN} of the {budget} budget ceiling",
                "threshold": f"total strength >= budget - {SEVERITY_BUDGET_MARGIN}",
                "valid": True})
        if recipe.geometry.shape == "boundary-only" and set(recipe.regions) <= {"face_boundary", "context"} \
                and len(recipe.regions) == 1 and recipe.regions == ["context"]:
            flags.append({
                "flag": "ODD_BUT_VALID", "recipe_id": recipe.recipe_id,
                "detail": "a boundary-only geometry covering only 'context' touches no facial region",
                "threshold": "geometry=boundary-only and regions==['context']",
                "valid": True})

    return flags


# -------------------------------------------------------------- C3 readiness
def c3_readiness(statistics: dict[str, Any], *, candidate_slots: int,
                 recipes_per_slot: int = 1) -> dict[str, Any]:
    """Project the observed C2 rates onto the planned C3 slot budget.

    These are projections from a 32-slot sample, not guarantees. The sample size
    is carried in the artifact so nobody reads a rate measured on 32 slots as a
    precise expectation for 384.
    """
    rates = statistics["rates"]
    slot_count = statistics["slots"]["slot_count"]
    calls_per_slot = statistics["calls"]["calls_per_slot_mean"]
    eventual = rates["eventual_valid_rate"]
    latency = statistics["latency_seconds"]
    tokens = statistics["token_usage"]

    expected_calls = round(candidate_slots * calls_per_slot, 2)
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "basis": {
            "observed_slots": slot_count,
            "observed_calls": statistics["calls"]["total_provider_calls"],
            "first_attempt_valid_rate": rates["first_attempt_valid_rate"],
            "eventual_valid_rate": eventual,
            "retry_rate": rates["retry_rate"],
            "duplicate_rate": rates["duplicate_rate"],
            "calls_per_slot_mean": calls_per_slot,
            "sample_size_caveat": f"every projection below extrapolates from {slot_count} slots; "
                                  "a rate measured on a sample this small carries wide uncertainty "
                                  "and is a planning input, not a guarantee",
        },
        "projection": {
            "candidate_slots": candidate_slots,
            "recipes_per_slot": recipes_per_slot,
            "expected_valid_candidates": round(candidate_slots * recipes_per_slot * eventual, 2),
            "expected_api_calls": expected_calls,
            "expected_retry_calls": round(expected_calls - candidate_slots, 2),
            "expected_retry_burden_calls_per_slot": round(max(calls_per_slot - 1.0, 0.0), 6),
            "expected_duplicate_candidates": round(candidate_slots * recipes_per_slot
                                                   * rates["duplicate_rate"], 2),
            "expected_exhausted_slots": round(candidate_slots * rates["retry_exhaustion_rate"], 2),
            "expected_wall_clock_seconds_serial": round(expected_calls * (latency["mean"] or 0.0), 2),
        },
        "quota_risk": {
            "observed_quota_exhaustion_events": statistics["counts"]["quota_exhausted"],
            "observed_transient_429": statistics["counts"]["transient_rate_limit_429"],
            "expected_calls_vs_free_tier": "the Free Tier requests-per-day limit for the frozen model "
                                           "is a project-specific number that must be read from AI "
                                           "Studio; it is not hard-coded here and was not invented",
            "expected_token_usage": tokens["totals"] if tokens["available"] else
                                    "not reported by the surface on these attempts",
            "policy": "on quota_exceeded the run checkpoints and stops; code never enables billing",
        },
    }


__all__ = ["AUDIT_SCHEMA_VERSION", "COVERAGE_AXES", "axis_categories", "coverage_audit",
           "cooccurrence_audit", "duplicate_audit", "structural_pattern", "pilot_statistics",
           "latency_stats", "token_totals", "percentile", "flag_pilot_issues", "c3_readiness"]
