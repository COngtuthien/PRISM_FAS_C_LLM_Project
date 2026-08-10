"""The prompt contract.

The system instruction and the generation template are built deterministically
from the frozen ontology, so the text the model sees always matches the validator
that will judge it. Both are hashed; the hashes enter the request identity.

This is the C1 template. It is NOT the C3 prompt lock: C2 may still tune wording
using source-only validity and coverage evidence, and the final bytes are frozen
in the LLM bank lock before the 384-slot scientific generation begins
(spec section 7.5).

Nothing here may reference a dataset, a target metric, a target failure or any
Version-B result. `build_generation_prompt` runs the request firewall over its
own output before returning it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from prism_fas.recipes.ontology import Ontology

from .firewall import assert_request_is_target_free

PROMPT_TEMPLATE_VERSION = "c1-llm-prompt-v1"


def _bullet(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join(values)


def _ranges_block(title: str, ranges: dict[str, Any]) -> str:
    lines = [f"{title}:"]
    for key in sorted(ranges):
        band = ranges[key]
        lines.append(f"  - {key}: {band.minimum} to {band.maximum}")
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_instruction: str
    generation_template: str
    ontology_identity: str

    @property
    def system_instruction_sha256(self) -> str:
        return hashlib.sha256(self.system_instruction.encode("utf-8")).hexdigest()

    @property
    def generation_template_sha256(self) -> str:
        return hashlib.sha256(self.generation_template.encode("utf-8")).hexdigest()

    def identity(self) -> str:
        return prompt_template_identity(self)

    def as_provenance(self) -> dict[str, Any]:
        return {"prompt_template_version": self.version,
                "system_prompt_sha256": self.system_instruction_sha256,
                "request_template_sha256": self.generation_template_sha256,
                "ontology_identity": self.ontology_identity,
                "prompt_template_identity": self.identity()}


def prompt_template_identity(template: PromptTemplate) -> str:
    material = {"version": template.version,
                "system_instruction_sha256": template.system_instruction_sha256,
                "generation_template_sha256": template.generation_template_sha256,
                "ontology_identity": template.ontology_identity}
    text = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_system_instruction(ontology: Ontology) -> str:
    """Role, no-target rule, schema, ontology, compatibility, hard output format,
    maximum artifacts/regions, forbidden shortcuts and an explicit JSON-only
    requirement (spec section 7.5)."""
    limits = ontology.limits
    compat_medium = "\n".join(
        f"  - {family}: {_bullet(ontology.artifacts_for_medium(family))}"
        for family in ontology.media)
    compat_geometry = "\n".join(
        f"  - {shape}: {_bullet(ontology.regions_for_geometry(shape))}"
        for shape in ontology.geometry_shapes)
    strengths = "\n".join(
        f"  - {name}: {ontology.strength_range(name).minimum} to {ontology.strength_range(name).maximum}"
        for name in ontology.artifacts)

    return f"""You are a constrained planner for physical presentation-attack artifact recipes in
face anti-spoofing research. You plan a structured artifact program. You do not
see images, you do not produce images, and you do not classify anything.

Your only output is JSON conforming to the provided response schema. Emit no
prose, no explanation, no markdown, no code fences. JSON only.

WHAT A RECIPE IS
A recipe describes, in generic physical terms, how a presentation medium held in
front of a camera would alter the appearance of a face: what the medium is made
of, what shape it takes, which facial regions it covers, which optical artifacts
it produces and at what severity, and under what capture conditions.

HARD RULES
1. Use only the vocabulary listed below. Never invent a medium, geometry, region,
   artifact, illumination value or route. A value outside these lists makes the
   whole recipe invalid.
2. Never invent a field. The response schema is exhaustive.
3. Do not emit a recipe id, a hash, a provenance block or a validity claim. Those
   are assigned by the system, and any value you supply for them is discarded.
4. Respect every numeric range. Values outside a range make the recipe invalid.
5. Total artifact strength across a recipe must not exceed
   {limits['max_total_artifact_strength']}.
6. Between {limits['min_artifacts_per_recipe']} and {limits['max_artifacts_per_recipe']} artifacts, and between
   {limits['min_regions_per_recipe']} and {limits['max_regions_per_recipe']} semantic regions, per recipe.
   No repeats within a recipe.
7. Regions must be listed in the canonical order given below.
8. Artifacts must be physically consistent with the medium, and regions must be
   reachable by the geometry, according to the compatibility tables below.
9. Aim for physical plausibility: choose combinations a real medium could
   actually produce, not arbitrary mixtures.
10. Aim for diversity across the batch: vary medium, geometry, region coverage,
    artifact composition, severity and capture conditions. Do not emit near
    copies of the same recipe.
11. Never make one artifact a universal shortcut. If a recipe declares a
    forbidden shortcut, that recipe must not consist solely of that artifact.

SCOPE RULE
You are given a generic ontology and nothing else. You have no information about
any corpus, benchmark, evaluation split, measurement or prior experiment, and you
must not speculate about, request or refer to any. Plan only from physics and
from the vocabulary below.

VOCABULARY
media: {_bullet(ontology.media)}
geometry shapes: {_bullet(ontology.geometry_shapes)}
semantic regions in canonical order: {_bullet(ontology.regions)}
artifacts: {_bullet(ontology.artifacts)}
illumination: {_bullet(ontology.illumination)}
generator routes: {_bullet(ontology.routes)}
forbidden shortcut policies: {_bullet(ontology.forbidden_shortcuts)}

COMPATIBILITY - artifacts each medium can produce:
{compat_medium}

COMPATIBILITY - regions each geometry can cover:
{compat_geometry}

ARTIFACT SEVERITY RANGES:
{strengths}

{_ranges_block("MEDIUM RANGES", ontology.medium_ranges)}

{_ranges_block("GEOMETRY RANGES", ontology.geometry_ranges)}

{_ranges_block("CAPTURE RANGES", ontology.capture_ranges)}

seed: an integer between {limits['seed_min']} and {limits['seed_max']}, chosen
freely and varied across recipes.

OUTPUT
A single JSON object with one key, "recipes", whose value is an array containing
exactly the number of recipe objects requested. Nothing else."""


def build_generation_template(ontology: Ontology) -> str:
    """The per-request body. Placeholders are filled by `build_generation_prompt`.

    It may carry only ontology-level coverage quotas and the batch size. It must
    never carry corpus-derived deficits or any feedback from an evaluation
    (spec section 7.5).
    """
    return """Produce exactly {recipes_requested} artifact recipes as one JSON object of the
form {{"recipes": [ ... ]}}.

Batch composition requirements, expressed only over the frozen vocabulary:
{coverage_block}

Within this batch, vary medium family, geometry shape, region coverage, artifact
composition, severity and capture conditions. Physical plausibility first,
diversity second, and never a repeated recipe.

Return JSON only."""


def _coverage_block(coverage_quotas: dict[str, Any] | None) -> str:
    if not coverage_quotas:
        return "  - none beyond the standing rules in the system instruction"
    lines = []
    for key in sorted(coverage_quotas):
        value = coverage_quotas[key]
        rendered = ", ".join(str(item) for item in value) if isinstance(value, (list, tuple)) else str(value)
        lines.append(f"  - {key}: {rendered}")
    return "\n".join(lines)


def build_prompt_template(ontology: Ontology) -> PromptTemplate:
    return PromptTemplate(
        version=PROMPT_TEMPLATE_VERSION,
        system_instruction=build_system_instruction(ontology),
        generation_template=build_generation_template(ontology),
        ontology_identity=ontology.sha256,
    )


# `load_prompt_template` is the name the rest of Version C imports, so a future
# milestone can swap a built template for frozen bytes read from the bank lock
# without touching callers.
def load_prompt_template(ontology: Ontology) -> PromptTemplate:
    return build_prompt_template(ontology)


def build_generation_prompt(template: PromptTemplate, *, recipes_requested: int,
                            coverage_quotas: dict[str, Any] | None = None) -> str:
    """Render the request body and prove it is target-free before it is used."""
    if not isinstance(recipes_requested, int) or isinstance(recipes_requested, bool) or recipes_requested < 1:
        raise ValueError("recipes_requested must be a positive integer")
    text = template.generation_template.format(
        recipes_requested=recipes_requested,
        coverage_block=_coverage_block(coverage_quotas),
    )
    # Fail closed on our own output: a coverage quota is the one caller-supplied
    # input to the prompt, and it must not smuggle anything target-shaped in.
    assert_request_is_target_free({"system_instruction": template.system_instruction,
                                   "input_text": text})
    return text
