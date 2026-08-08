"""The A02 control artifact: a RANDOM-OPERATOR recipe bank, source-only.

Table 60's recipe ablation asks "random operators vs structured recipe", and the
spec's hypothesis H4 fixes the protocol:

    "Structured recipe bank tot hon random augmentation cung so sample va cung
     detector."

Same sample count, same detector. So the control must hold constant everything
except HOW the operator composition was chosen, and the one thing it may not do is
reuse the structured payloads under permuted recipe ids — that changes the
conditioning LABEL while leaving the data identical, which tests nothing the spec
asks about.

## What is held constant

```
operator set        the same M7 ontology vocabulary and the same operator code
parameter bands     the same ontology ranges — these are OPERATOR SAFETY limits,
                    not composition rules, and running outside them would make the
                    comparison about numerical stability instead of composition
severity budget     the same ontology total-strength limit, for the same reason
recipe count        the same 128
live targets        the same source_train live population
routes              the same physics/GPAT mix and the same candidate-plan shape
generator           the same frozen PhysicsEngine and the same frozen GPAT checkpoint
quality gate        the SAME frozen M8 v3 thresholds; acceptance is not rebuilt
```

## What is randomized — the composition semantics, and only that

```
medium/artifact compatibility   DROPPED: artifacts are drawn from the FULL
                                vocabulary, not from `artifacts_for_medium`
geometry/region compatibility   DROPPED: regions are drawn from the FULL region
                                list, not from `regions_for_geometry`
coverage strides                DROPPED: the structured generator advances the
                                categorical axes by fixed strides so that coverage
                                is a property of the construction; here every
                                categorical axis is an independent uniform draw
severity allocation             DROPPED: the structured generator distributes one
                                sampled total across the artifacts; here each
                                artifact's strength is an independent uniform draw
                                inside its own band, then scaled down only if the
                                shared budget would be exceeded
```

That is exactly "a random composition of the same operators" against "a physically
coherent composition of the same operators".

## Identity

The bank writes the same file set as an M7 bank so `recipes.bank.load_bank` and the
whole M8 generation pipeline read it unchanged, but its content identity binds the
RANDOM policy and its own version string. The two banks can therefore never be
confused, and a bank built by this module can never claim the M7 identity.

Source-only by construction: this module reads the ontology and nothing else. No
target path, no target taxonomy, no label, no image.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
from prism_fas.recipes.canonical import canonical_json, recipe_hash, text_hash
from prism_fas.recipes.compile import COMPILER_VERSION, compile_recipes, compile_summary, derive_seed, local_rng
from prism_fas.recipes.conditioning import CONDITIONING_DIM, CONDITIONING_VERSION, feature_names_sha256
from prism_fas.recipes.generate import recipes_to_jsonl
from prism_fas.recipes.ontology import Ontology, load_ontology
from prism_fas.recipes.schema import RecipeV11, parse_recipe

RANDOM_BANK_SCHEMA_VERSION = "m10-random-operator-bank-v1"
RANDOM_BANK_STATUS_FROZEN = "frozen"
RANDOM_POLICY_VERSION = "m10-uniform-operator-composition-v1"
# The A02 control artifact id. Independent and immutable; never an M7/M8 id.
DEFAULT_BANK_ID = "prism_recipe_bank_m10_random_v1"
DEFAULT_RECIPE_COUNT = 128
DEFAULT_BANK_SEED = 20260806
# The same file set an M7 bank writes, so `load_bank` reads this one unchanged.
BANK_FILES = ("recipes.jsonl", "ontology.yaml", "prompt.txt", "generator.json", "coverage.json",
              "validation.json", "BANK_LOCK.json")
CONTENT_FILES = ("recipes.jsonl", "ontology.yaml", "prompt.txt", "generator.json", "coverage.json",
                 "validation.json")


class RandomOperatorBankError(RuntimeError):
    """The A02 control bank cannot be built or is being overwritten."""


def _round(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals) + 0.0


def _uniform_strengths(names: list[str], ontology: Ontology, rng: np.random.Generator) -> list[float]:
    """Each artifact's strength drawn INDEPENDENTLY inside its own safe band.

    The structured generator samples one total severity and distributes it across
    the artifacts, which is a composition decision. Here there is no coordination:
    each draw is independent. The shared ontology budget is still a hard safety
    limit for both arms, so if independent draws exceed it they are scaled down
    proportionally rather than re-allocated — the scaling preserves the ratio the
    random draw produced instead of imposing a structured one.
    """
    import math
    budget = float(ontology.limits["max_total_artifact_strength"])
    bands = [ontology.strength_range(name) for name in names]
    values = [_round(float(rng.uniform(band.minimum, band.maximum))) for band in bands]
    total = sum(values)
    if total > budget + 1e-9:
        # Rounding must only ever REDUCE here. Scaling to exactly the budget and then
        # rounding to 4 decimals can round three values up and put the sum back over
        # it, which is a rounding artifact rather than an infeasible combination —
        # every band minimum in this ontology is 0.05, so the floor of any 3-artifact
        # set is 0.15 and no set is genuinely infeasible against a budget of 1.0.
        scale = budget / total
        floor4 = lambda value: math.floor(float(value) * 10_000) / 10_000
        values = [max(band.minimum, floor4(value * scale)) + 0.0
                  for value, band in zip(values, bands)]
    if sum(values) > budget + 1e-9:
        # Only reachable if an ontology is ever revised so that the band minima alone
        # exceed the budget. Refuse rather than silently re-allocate into a structured
        # distribution, which is the very thing this bank ablates.
        raise RandomOperatorBankError(
            f"artifact set {names} cannot fit the severity budget {budget}: band minima sum to "
            f"{sum(band.minimum for band in bands)}")
    return values


def build_random_recipe(index: int, ontology: Ontology, *, bank_id: str, bank_seed: int) -> RecipeV11:
    """One recipe whose composition is a uniform draw over the operator vocabulary.

    Deterministic: every value comes from a recipe-local PCG64 seeded through the
    same SHA-256 derivation the compiler uses. No global RNG is touched, so the
    bank is reproducible on any machine.
    """
    recipe_id = f"R-{index + 1:06d}"
    rng = local_rng(derive_seed(bank_id, recipe_id, bank_seed, RANDOM_POLICY_VERSION, index))
    max_artifacts = int(ontology.limits["max_artifacts_per_recipe"])
    max_regions = int(ontology.limits["max_regions_per_recipe"])

    # Categorical axes: independent uniform draws. No stride, no coverage guarantee,
    # and deliberately NO medium/artifact or geometry/region compatibility filter —
    # those filters ARE the structured composition being ablated.
    family = str(ontology.media[int(rng.integers(len(ontology.media)))])
    shape = str(ontology.geometry_shapes[int(rng.integers(len(ontology.geometry_shapes)))])
    illumination = str(ontology.illumination[int(rng.integers(len(ontology.illumination)))])
    artifact_count = 1 + int(rng.integers(max_artifacts))
    region_count = 1 + int(rng.integers(max_regions))
    names = [str(ontology.artifacts[index_]) for index_ in
             rng.choice(len(ontology.artifacts), size=artifact_count, replace=False)]
    regions = ontology.sort_regions([str(ontology.regions[index_]) for index_ in
                                     rng.choice(len(ontology.regions), size=region_count, replace=False)])
    strengths = _uniform_strengths(names, ontology, rng)

    medium_ranges, geometry_ranges, capture = (ontology.medium_ranges, ontology.geometry_ranges,
                                               ontology.capture_ranges)
    # `physics` is mandatory for a compilable graph; the GPAT half of the route mix
    # follows the same 50/50 split the structured bank uses, because the ROUTE is an
    # A03 dimension and must not move here.
    routes = ["physics", "gpat"] if index % 2 == 0 else ["physics"]
    payload: dict[str, Any] = {
        "recipe_id": recipe_id,
        "medium": {"family": family,
                   "transparency": _round(rng.uniform(medium_ranges["transparency"].minimum,
                                                      medium_ranges["transparency"].maximum)),
                   "roughness": _round(rng.uniform(medium_ranges["roughness"].minimum,
                                                   medium_ranges["roughness"].maximum))},
        "geometry": {"shape": shape,
                     "rigidity": _round(rng.uniform(geometry_ranges["rigidity"].minimum,
                                                    geometry_ranges["rigidity"].maximum)),
                     "coverage": _round(rng.uniform(geometry_ranges["coverage"].minimum,
                                                    geometry_ranges["coverage"].maximum))},
        "regions": regions,
        "artifacts": [{"name": name, "strength": strength} for name, strength in zip(names, strengths)],
        "capture": {"yaw": _round(rng.uniform(capture["yaw"].minimum, capture["yaw"].maximum), 1),
                    "illumination": illumination,
                    "compression_q": int(rng.integers(int(capture["compression_q"].minimum),
                                                      int(capture["compression_q"].maximum) + 1)),
                    "scale": _round(rng.uniform(capture["scale"].minimum, capture["scale"].maximum), 3),
                    "motion": _round(rng.uniform(capture["motion"].minimum, capture["motion"].maximum), 3),
                    "defocus": _round(rng.uniform(capture["defocus"].minimum, capture["defocus"].maximum), 3)},
        # The shortcut guards are a SAFETY declaration, not a composition rule, and
        # are carried unchanged so both arms forbid the same degenerate recipes.
        "forbidden_shortcuts": list(ontology.forbidden_shortcuts),
        "generator_route": routes,
        "seed": int(rng.integers(int(ontology.limits["seed_min"]), int(ontology.limits["seed_max"]))),
        "schema_version": ontology.recipe_schema_version}
    return parse_recipe(payload)


def generate_random_recipes(ontology: Ontology, *, count: int, bank_id: str,
                            bank_seed: int) -> list[RecipeV11]:
    """`count` recipes, all structurally valid, none ontology-composed.

    The only property asserted here is that the canonical hashes are distinct: a
    duplicate would silently shrink the effective bank and make the two arms differ
    in diversity as well as in composition policy. Coverage is deliberately NOT
    asserted — an enforced coverage guarantee is itself a structural property.
    """
    recipes = [build_random_recipe(index, ontology, bank_id=bank_id, bank_seed=bank_seed)
               for index in range(int(count))]
    digests = [recipe_hash(recipe) for recipe in recipes]
    duplicates = sorted({value for value in digests if digests.count(value) > 1})
    if duplicates:
        raise RandomOperatorBankError(f"the random generator produced duplicate canonical hashes: {duplicates}")
    return recipes


def composition_report(recipes: list[RecipeV11], ontology: Ontology) -> dict[str, Any]:
    """What the random policy actually produced, MEASURED rather than asserted.

    `off_manifold_*` counts the recipes whose (medium, artifact) or
    (geometry, region) pairing the structured ontology would have refused. It is
    the direct evidence that this bank is not a structured bank with a different
    name; a value of zero would mean the ablation changed nothing.
    """
    off_medium = off_geometry = 0
    for recipe in recipes:
        allowed_artifacts = set(ontology.artifacts_for_medium(recipe.medium.family))
        allowed_regions = set(ontology.regions_for_geometry(recipe.geometry.shape))
        if any(spec.name not in allowed_artifacts for spec in recipe.artifacts): off_medium += 1
        if any(name not in allowed_regions for name in recipe.regions): off_geometry += 1
    counter = lambda values: {str(key): values.count(key) for key in sorted(set(values))}
    return {
        "recipes": len(recipes),
        "media": counter([recipe.medium.family for recipe in recipes]),
        "geometry_shapes": counter([recipe.geometry.shape for recipe in recipes]),
        "illumination": counter([recipe.capture.illumination for recipe in recipes]),
        "artifacts": counter([spec.name for recipe in recipes for spec in recipe.artifacts]),
        "regions": counter([name for recipe in recipes for name in recipe.regions]),
        "artifact_counts": counter([len(recipe.artifacts) for recipe in recipes]),
        "region_counts": counter([len(recipe.regions) for recipe in recipes]),
        "routes": counter(["+".join(recipe.generator_route) for recipe in recipes]),
        "total_strength": {
            "min": round(min(sum(s.strength for s in r.artifacts) for r in recipes), 6),
            "max": round(max(sum(s.strength for s in r.artifacts) for r in recipes), 6),
            "budget": float(ontology.limits["max_total_artifact_strength"])},
        "off_manifold_medium_artifact": off_medium,
        "off_manifold_geometry_region": off_geometry,
        "off_manifold_any": sum(1 for recipe in recipes
                                if any(spec.name not in set(ontology.artifacts_for_medium(recipe.medium.family))
                                       for spec in recipe.artifacts)
                                or any(name not in set(ontology.regions_for_geometry(recipe.geometry.shape))
                                       for name in recipe.regions)),
        "every_recipe_declares_physics": all("physics" in recipe.generator_route for recipe in recipes)}


def random_bank_content_identity(*, bank_id: str, recipe_count: int, bank_seed: int,
                                 file_hashes: dict[str, str], recipe_hashes: dict[str, str],
                                 graph_hashes: dict[str, str], ontology_version: str,
                                 conditioning: dict[str, Any]) -> str:
    """Identity over content only, binding the RANDOM policy version.

    No wall-clock, host, machine path or build duration participates, so a rebuild
    on another machine reproduces it. The policy version is inside the hash, so a
    structured bank and a random bank can never collide even at equal content.
    """
    payload = {"bank_schema_version": RANDOM_BANK_SCHEMA_VERSION,
               "composition_policy": RANDOM_POLICY_VERSION,
               "bank_id": bank_id, "recipe_count": int(recipe_count), "bank_seed": int(bank_seed),
               "file_sha256": dict(sorted(file_hashes.items())),
               "recipe_hashes": dict(sorted(recipe_hashes.items())),
               "graph_hashes": dict(sorted(graph_hashes.items())),
               "ontology_version": ontology_version, "compiler_version": COMPILER_VERSION,
               "conditioning": dict(sorted(conditioning.items()))}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _policy_text(ontology: Ontology) -> str:
    """The frozen statement of what this bank is, written beside it.

    An M7 bank ships `prompt.txt`, the constrained contract a generator is held to.
    This bank's contract is a sampling policy, so that is what its `prompt.txt`
    contains — including, explicitly, what it does NOT constrain.
    """
    return "\n".join([
        "PRISM-FAS-B M10 A02 control: uniform random operator composition",
        f"composition_policy: {RANDOM_POLICY_VERSION}",
        f"ontology_version: {ontology.version}",
        f"ontology_sha256: {ontology.sha256}",
        f"recipe_schema_version: {ontology.recipe_schema_version}",
        "",
        "PURPOSE",
        "The control arm of Table 60's recipe ablation and of hypothesis H4:",
        "'structured recipe bank vs random augmentation at equal sample count and",
        "equal detector'. It exists to be compared against the frozen M7 structured",
        "bank, and it is never a substitute for it.",
        "",
        "HELD CONSTANT (identical to the structured arm)",
        f"  operator vocabulary      {list(ontology.artifacts)}",
        f"  region vocabulary        {list(ontology.regions)}",
        "  per-artifact safe bands  the ontology's own ranges (operator safety limits)",
        f"  total severity budget    {ontology.limits['max_total_artifact_strength']}",
        f"  max artifacts / regions  {ontology.limits['max_artifacts_per_recipe']} / "
        f"{ontology.limits['max_regions_per_recipe']}",
        "  forbidden shortcuts      carried unchanged",
        "  recipe schema            unchanged, so the same compiler and the same",
        "                           physics/GPAT routes consume it unchanged",
        "",
        "DELIBERATELY NOT CONSTRAINED (this is the ablation)",
        "  medium / artifact compatibility   artifacts are drawn from the FULL",
        "                                    vocabulary, not from the medium's",
        "                                    physically compatible subset",
        "  geometry / region compatibility   regions are drawn from the FULL list,",
        "                                    not from the geometry's compatible subset",
        "  categorical coverage              no stride, no guarantee; every axis is",
        "                                    an independent uniform draw",
        "  severity allocation               independent per-artifact draws, not one",
        "                                    total distributed across the artifacts",
        "",
        "SOURCE-ONLY",
        "No evaluation/target dataset, attack taxonomy, file path, subject identity",
        "or model performance number is read, named or inferred anywhere in this",
        "bank's construction. It is a function of the ontology and the seed alone.",
        ""])


def build_random_operator_bank(output_root: Path, ontology_path: Path, *,
                               bank_id: str = DEFAULT_BANK_ID,
                               recipe_count: int = DEFAULT_RECIPE_COUNT,
                               bank_seed: int = DEFAULT_BANK_SEED,
                               dry_run: bool = False) -> dict[str, Any]:
    """Build or reuse the frozen A02 control bank.

    Rebuilding with identical inputs is a no-op and the existing files are proven
    byte-identical. A destination holding a DIFFERENT lock is never overwritten:
    this artifact is immutable, exactly like the M7 and M8 banks it is compared
    against.
    """
    ontology = load_ontology(Path(ontology_path))
    recipes = generate_random_recipes(ontology, count=recipe_count, bank_id=bank_id, bank_seed=bank_seed)
    graphs = compile_recipes(recipes, ontology, bank_id=bank_id)
    compiled = compile_summary(graphs)
    if compiled["duplicate_graph_hashes"]:
        raise RandomOperatorBankError(f"compiled graphs collide: {compiled['duplicate_graph_hashes']}")
    report = composition_report(recipes, ontology)
    if not report["every_recipe_declares_physics"]:
        raise RandomOperatorBankError("every recipe must declare the physics route to be compilable")
    if report["off_manifold_any"] == 0:
        # If a uniform draw over the full vocabulary never leaves the structured
        # manifold, this bank is a structured bank with a different name and the
        # ablation measures nothing. Refuse rather than ship a null control.
        raise RandomOperatorBankError(
            "no recipe left the structured compatibility manifold; this would be a structured "
            "bank under another name and cannot serve as the A02 control")

    generator_manifest = {
        "provider": "deterministic_local", "model_id": "m10-random-operator-composition",
        "revision": RANDOM_POLICY_VERSION, "external_llm_invoked": False,
        "network_access": False, "credential_used": False,
        "bank_id": bank_id, "bank_seed": int(bank_seed), "requested_recipe_count": int(recipe_count),
        "ontology_version": ontology.version, "ontology_sha256": ontology.sha256,
        "recipe_schema_version": ontology.recipe_schema_version,
        "policy": "uniform draws over the ontology vocabulary and parameter bands; the medium/artifact "
                  "and geometry/region compatibility rules, the coverage strides and the coordinated "
                  "severity allocation are deliberately NOT applied",
        "seed_derivation": f"sha256(bank_id|recipe_id|bank_seed|'{RANDOM_POLICY_VERSION}'|index) -> PCG64",
        "purpose": "M10 A02 control for Table 60 recipe ablation / hypothesis H4",
        "source_only": True}
    validation = {"bank_id": bank_id, "passed": True,
                  "schema_validated": len(recipes),
                  "note": "structural schema validation only. The ontology COMPOSITION rules are "
                          "intentionally not enforced: they are the thing this bank ablates.",
                  "duplicate_recipe_hashes": [], "duplicate_graph_hashes": []}
    json_text = lambda payload: json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    texts = {"recipes.jsonl": recipes_to_jsonl(recipes),
             "ontology.yaml": ontology.source_text,
             "prompt.txt": _policy_text(ontology),
             "generator.json": json_text(generator_manifest),
             "coverage.json": json_text({"bank_id": bank_id, "composition": report, "compiled": compiled}),
             "validation.json": json_text(validation)}

    file_hashes = {name: text_hash(texts[name]) for name in CONTENT_FILES}
    recipe_hashes = {recipe.recipe_id: recipe_hash(recipe) for recipe in recipes}
    graph_hashes = {graph.recipe_id: graph.graph_hash for graph in graphs}
    conditioning = {"version": CONDITIONING_VERSION, "dimension": CONDITIONING_DIM,
                    "feature_names_sha256": feature_names_sha256(ontology)}
    identity = random_bank_content_identity(
        bank_id=bank_id, recipe_count=len(recipes), bank_seed=bank_seed, file_hashes=file_hashes,
        recipe_hashes=recipe_hashes, graph_hashes=graph_hashes, ontology_version=ontology.version,
        conditioning=conditioning)
    lock = {"bank_schema_version": RANDOM_BANK_SCHEMA_VERSION,
            "composition_policy": RANDOM_POLICY_VERSION,
            "bank_id": bank_id, "status": RANDOM_BANK_STATUS_FROZEN,
            "purpose": "M10 A02 control (Table 60 recipe ablation / hypothesis H4)",
            "recipe_count": len(recipes), "bank_seed": int(bank_seed),
            "generator": {key: generator_manifest[key] for key in
                          ("provider", "model_id", "revision", "external_llm_invoked")},
            "ontology_version": ontology.version, "ontology_sha256": file_hashes["ontology.yaml"],
            "prompt_sha256": file_hashes["prompt.txt"],
            "recipes_jsonl_sha256": file_hashes["recipes.jsonl"],
            "coverage_sha256": file_hashes["coverage.json"],
            "validation_sha256": file_hashes["validation.json"],
            "generator_sha256": file_hashes["generator.json"], "file_sha256": file_hashes,
            "compiler_version": COMPILER_VERSION, "conditioning": conditioning,
            "recipe_hashes": recipe_hashes, "graph_hashes": graph_hashes,
            "composition_report": report,
            "source_only": True, "target_paths": 0, "target_taxonomy": 0,
            "bank_content_identity_sha256": identity}
    texts_with_lock = {**texts, "BANK_LOCK.json": json_text(lock)}

    root = Path(output_root)
    plan = {"bank_id": bank_id, "recipe_count": len(recipes), "bank_seed": int(bank_seed),
            "output_root_name": root.name, "ontology_version": ontology.version,
            "bank_content_identity_sha256": identity, "files": list(BANK_FILES),
            "composition": report, "compiled": compiled}
    existing = root / "BANK_LOCK.json"
    if existing.is_file():
        recorded = json.loads(existing.read_text(encoding="utf-8"))
        if recorded.get("bank_content_identity_sha256") != identity:
            raise RandomOperatorBankError(
                f"{root.name} already holds bank {recorded.get('bank_id')!r} with identity "
                f"{recorded.get('bank_content_identity_sha256')} != {identity}; this artifact is "
                "immutable, create a new versioned directory instead")
        differing = [name for name in BANK_FILES if not (root / name).is_file()
                     or (root / name).read_text(encoding="utf-8") != texts_with_lock[name]]
        if differing:
            raise RandomOperatorBankError(
                f"{root.name} has a matching lock but differing files {differing}; refusing to rewrite")
        return {"status": "reused", "written": [], **plan}
    if dry_run: return {"status": "dry_run", "written": [], **plan}
    root.mkdir(parents=True, exist_ok=True)
    for name in BANK_FILES:
        _atomic_text(root / name, texts_with_lock[name])
    return {"status": "created", "written": list(BANK_FILES), **plan}


def _atomic_text(path: Path, content: str) -> None:
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def validate_random_operator_bank(root: Path) -> dict[str, Any]:
    """Re-derive every hash in BANK_LOCK.json from the files on disk."""
    from prism_fas.recipes.bank import load_bank
    root = Path(root)
    missing = [name for name in BANK_FILES if not (root / name).is_file()]
    if missing: raise RandomOperatorBankError(f"{root.name} is not a random-operator bank; missing {missing}")
    bank = load_bank(root)
    lock, ontology, recipes = bank["lock"], bank["ontology"], bank["recipes"]
    errors: list[str] = []
    if lock.get("bank_schema_version") != RANDOM_BANK_SCHEMA_VERSION:
        errors.append(f"bank_schema_version {lock.get('bank_schema_version')!r} is not this artifact's")
    if lock.get("composition_policy") != RANDOM_POLICY_VERSION: errors.append("composition_policy mismatch")
    if lock.get("status") != RANDOM_BANK_STATUS_FROZEN: errors.append("status is not frozen")
    if int(lock.get("recipe_count", -1)) != len(recipes): errors.append("recipe_count mismatch")
    file_hashes = {name: text_hash((root / name).read_text(encoding="utf-8")) for name in CONTENT_FILES}
    for name, digest in file_hashes.items():
        if lock.get("file_sha256", {}).get(name) != digest: errors.append(f"{name} sha256 mismatch")
    graphs = compile_recipes(recipes, ontology, bank_id=str(lock["bank_id"]))
    recipe_hashes = {recipe.recipe_id: recipe_hash(recipe) for recipe in recipes}
    graph_hashes = {graph.recipe_id: graph.graph_hash for graph in graphs}
    if lock.get("recipe_hashes") != recipe_hashes: errors.append("recipe_hashes mismatch")
    if lock.get("graph_hashes") != graph_hashes: errors.append("graph_hashes mismatch")
    conditioning = {"version": CONDITIONING_VERSION, "dimension": CONDITIONING_DIM,
                    "feature_names_sha256": feature_names_sha256(ontology)}
    if lock.get("conditioning") != conditioning: errors.append("conditioning contract mismatch")
    identity = random_bank_content_identity(
        bank_id=str(lock["bank_id"]), recipe_count=len(recipes), bank_seed=int(lock["bank_seed"]),
        file_hashes=file_hashes, recipe_hashes=recipe_hashes, graph_hashes=graph_hashes,
        ontology_version=ontology.version, conditioning=conditioning)
    if lock.get("bank_content_identity_sha256") != identity:
        errors.append("bank_content_identity_sha256 mismatch")
    report = composition_report(recipes, ontology)
    if lock.get("composition_report") != report: errors.append("composition_report mismatch")
    if report["off_manifold_any"] == 0: errors.append("the bank never leaves the structured manifold")
    return {"bank_root_name": root.name, "bank_id": str(lock["bank_id"]), "passed": not errors,
            "errors": errors, "recipe_count": len(recipes),
            "bank_content_identity_sha256": identity,
            "composition_policy": lock.get("composition_policy"),
            "composition_report": report, "compiled": compile_summary(graphs),
            "ontology_version": ontology.version, "source_only": bool(lock.get("source_only"))}
