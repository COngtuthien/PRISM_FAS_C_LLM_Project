"""PRISM-FAS-C EXT-Q1Q2 - E6: LLM-SHUFFLE-A render adapter.

Historical Track-G training never reads recipes directly: it reads a
RENDERED, quality-gated, C6-matched candidate bank
(`detector.c6_bank.open_arm_bank` -> `C6MatchedBankReader` ->
`detector.dataset.M9TrainingDataset`). Because a recipe's exact parameter
combination determines the rendered pixels, LLM-SHUFFLE-A's 256
field-shuffled recipes need their OWN render -- the historical LLM rendered
bank cannot be substituted without defeating the experiment (see
`E6_RENDER_HISTORICAL_PARITY_AUDIT.json`).

This module is a SMALL ADDITIVE ADAPTER, not a fork of the renderer. It
reuses, unmodified, by direct import:

* `synthesis.c5_source_pair_plan` -- the FROZEN, arm-independent base
  schedule (`C5_SOURCE_PAIR_PLAN.json`'s `positions`: which live image, which
  route, which recipe ordinal, at every one of the 2048 slots). Read
  read-only; never rebuilt.
* `synthesis.c5_raw_generation.GenerationIdentity` / `CandidateRecord` /
  `candidate_dir` / `write_record` / `read_record` / `reuse_decision` -- the
  EXACT per-candidate identity, on-disk layout and resume-safety machinery
  historical C5 rendering already uses (a candidate record is written LAST,
  atomically; its presence alone means the payloads are complete, so a
  resumed render already reuses valid candidates without rerendering them --
  this module adds nothing here, it only calls the existing functions).
* `synthesis.c5_render.build_routes` / `render_one` / `render_arm` -- the
  frozen physics/GPAT renderers, called with the SAME checkpoint/engine
  identities ORIGINAL_LLM used, against a `plan` dict this module builds
  (mirroring `c5_arm_plan.build_arm_plan`'s shape) for a NEW pseudo-arm,
  `"LLM_SHUFFLE_A"`.
* `detector.c6_bank.C6MatchedBankReader.open` -- proven, not merely assumed,
  to accept an arbitrary `arm` string with no closed vocabulary check (unlike
  `synthesis.c5_source_pair_plan.arm_candidate_plan_identity`, which is
  deliberately closed to `{RND, DET, LLM}` and is therefore NOT called here
  for the plan identity -- this module computes its own, analogous, E6-only
  plan identity instead, exactly because that frozen function's closed
  vocabulary must not be widened).

Everything else -- renderer identities, source image pool, candidate
multiplicity, quality model/thresholds, C6 matching policy, final bank size
-- is pinned to ORIGINAL_LLM's own real, frozen values and asserted equal,
never re-derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e6_training_plan as training_plan

E6_DIR = "reports/c_ext_q1q2_v1/e6_llm_shuffle"
RENDER_DIR = f"{E6_DIR}/render"
RENDER_PLAN_PATH = f"{RENDER_DIR}/E6_RENDER_PLAN.json"
RENDER_PLAN_LOCK_PATH = f"{RENDER_DIR}/E6_RENDER_PLAN_LOCK.json"
RENDER_PARITY_AUDIT_PATH = f"{RENDER_DIR}/E6_RENDER_PARITY_AUDIT.json"
RENDER_PROVENANCE_PATH = f"{RENDER_DIR}/E6_RENDER_PROVENANCE.json"
HISTORICAL_PARITY_AUDIT_PATH = f"{RENDER_DIR}/E6_RENDER_HISTORICAL_PARITY_AUDIT.json"

#: Additive-only. Never `reports/full/c5`, `reports/full/c6`,
#: `reports/c3/scientific`, or `assets/recipe_banks/c3` -- those are historical
#: and frozen.
CANDIDATES_ROOT = "runs/c_ext_q1q2_v1/EXT-F1/e6_llm_shuffle/render/candidates"
MATCHED_BANK_ROOT = "data/processed/c_ext_q1q2_v1/e6_llm_shuffle/matched_bank"
BANK_LOCK_PATH = f"{MATCHED_BANK_ROOT}/E6_SHUFFLE_A_BANK_LOCK.json"
SOURCE_PAIR_ALIGNMENT_LOCK_PATH = f"{RENDER_DIR}/E6_SOURCE_PAIR_ALIGNMENT_LOCK.json"
Q_AUDIT_PATH = f"{RENDER_DIR}/E6_Q_AUDIT.json"

#: The canonical, already-established locators the historical C5 orchestration
#: (`pipeline.adapters.c5._render_candidates` / `_verify_c4_lock`) itself
#: resolves from -- reused verbatim, never a laptop-only literal.
C4_SCIENTIFIC_LOCK_PATH = "reports/full/c4/GPAT_CONFIG_LOCK.json"
SOURCE_PACKAGE_ROOT = "data/packages/prism_data_v1_m3b"
SOURCE_TRAIN_MANIFEST_RELATIVE = "manifests/source_train.parquet"
ONTOLOGY_CONFIG_PATH = "configs/recipes/ontology_m7.yaml"

#: The REAL, frozen quality calibration `pipeline.adapters.c6.FIT_NOMINAL_CALIBRATION`
#: wrote (`FrozenCalibration.load` reads exactly this shape). E6 binds its
#: `threshold_sha256` and must fail closed if it drifts from
#: `EXPECTED_QUALITY_THRESHOLD_IDENTITY` -- never refit, never substituted.
QUALITY_CALIBRATION_PATH = "reports/full/c6/QUALITY_CALIBRATION.json"
#: Frozen local weight root, matching `quality_models.PINNED`'s
#: `relative_path`/`alternate_relative_paths`. GPU-host-only on this laptop.
QUALITY_WEIGHT_ROOT = "weights"
#: Where the HISTORICAL ORIGINAL_LLM rendered candidate payload bytes live
#: (`pipeline.adapters.c5._scientific_work_root`: `runs/full/c5/scientific/candidates`).
#: Read-only, TASK G evidence only -- never written, never re-rendered.
HISTORICAL_LLM_CANDIDATE_ROOT = "runs/full/c5/scientific/candidates"

#: The already-frozen project-wide q-confound trigger (E2/E8), reused
#: verbatim -- this module only EMITS whether it would fire, never acts on it.
E8_SMD_TRIGGER_THRESHOLD = 0.25

E6_ARM_NAME = "LLM_SHUFFLE_A"

#: Historical ORIGINAL_LLM authoritative values, read directly from the real,
#: frozen artifacts this milestone's re-audit inspected -- pinned here so a
#: drifted historical lock is caught rather than silently trusted.
C5_ARM_PLANS_PATH = "reports/full/c5/C5_ARM_PLANS.json"
C5_SOURCE_PAIR_PLAN_PATH = "reports/full/c5/C5_SOURCE_PAIR_PLAN.json"
#: The ACTIVE (non-superseded) C5 lock -- `lock_kind="scientific_candidate_pool"`.
#: A prior, SUPERSEDED attempt is archived under `reports/full/c5/superseded/`
#: and is named by THIS file's own `supersedes.archived_lock` field when one
#: exists (never guessed from directory listing order or timestamps).
C5_SYNTHESIS_LOCK_PATH = "reports/full/c5/C5_SYNTHESIS_LOCK.json"
C6_BANK_LOCK_LLM_PATH = "reports/full/c6/C6_BANK_LOCK_LLM.json"
C6_GATE_PROFILES_PATH = "reports/full/c6/C6_GATE_PROFILES.json"
C3_BANK_LLM_PATH = "assets/recipe_banks/c3/llm/C3_BANK.json"
RECIPE_BANK_LLM_JSONL_PATH = "assets/recipe_banks/c3/llm/recipes.jsonl"

EXPECTED_ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"
EXPECTED_GPAT_CHECKPOINT_SHA256 = "80e852b3a0cd6eab9d31a05df0c9bc3dae53ce36dba9ede91d9b5ef365e3a755"
EXPECTED_PHYSICS_ENGINE_VERSION = "m7-physics-v1"
EXPECTED_SOURCE_PAIR_PLAN_IDENTITY = "75f31415f7205c0f760e6ce1f1fccdbc1431a8f73fc46e9a6cea388e2fdde515"
EXPECTED_PACKAGE_IDENTITY = "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"
EXPECTED_RENDERS_PER_RECIPE = 8
EXPECTED_CANDIDATES_PER_ARM = 2048
EXPECTED_QUALITY_PROFILE = "NOMINAL"
EXPECTED_QUALITY_THRESHOLD_IDENTITY = "8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19"
EXPECTED_FINAL_BANK_SIZE = 1024
EXPECTED_BY_ROUTE_QUOTA = {"physics": 512, "gpat": 512}
EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY = "fcc4c8005c0699c903909ab19bcc87800b73a2fc2d28d1a6eab73bcbd8a8f326"

PLAN_SCHEMA_VERSION = "e6-render-plan-v1"
LOCK_SCHEMA_VERSION = "e6-render-plan-lock-v1"

#: Fields that must be IDENTICAL between the ORIGINAL_LLM render plan and the
#: LLM-SHUFFLE-A render plan (TASK B). Deliberately excludes recipe/bank
#: identities, arm name and candidate ids, which are EXPECTED to differ.
PARITY_FIELDS = (
    "ontology_identity", "gpat_checkpoint_sha256", "physics_engine_version",
    "source_pair_plan_identity", "package_identity", "renders_per_recipe",
    "candidates_per_arm", "quality_profile", "quality_threshold_identity",
    "final_bank_size", "by_route_quota",
)


class E6RenderError(RuntimeError):
    """A precondition for the E6 render plan/execution failed. Fails closed."""


# --------------------------------------------------------------------------- #
# A. Historical ORIGINAL_LLM path re-audit (real artifacts only)
# --------------------------------------------------------------------------- #

def audit_historical_path(repo: Path) -> dict[str, Any]:
    arm_plans = cc.read_json(repo / C5_ARM_PLANS_PATH)
    llm_arm = arm_plans["arms"]["LLM"]
    source_pair_plan = cc.read_json(repo / C5_SOURCE_PAIR_PLAN_PATH)
    bank_lock = cc.read_json(repo / C6_BANK_LOCK_LLM_PATH)
    gate_profiles = cc.read_json(repo / C6_GATE_PROFILES_PATH)
    c3_bank = cc.read_json(repo / C3_BANK_LLM_PATH)

    values = {
        "source_recipe_path": "assets/recipe_banks/c3/llm/recipes.jsonl",
        "recipe_bank_identity": llm_arm["recipe_bank_identity"],
        "recipe_count": 256,
        "arm_plan_identity": llm_arm["arm_plan_identity"],
        "renders_per_recipe": source_pair_plan["renders_per_recipe"],
        "candidates_per_arm": source_pair_plan["candidates_per_arm"],
        "total_candidate_count": llm_arm["planned_candidates"],
        "gpat_checkpoint_sha256": arm_plans["gpat_checkpoint_sha256"],
        "physics_engine_version": arm_plans["physics_engine_version"],
        "package_identity": source_pair_plan["package_identity"],
        "source_pair_plan_identity": source_pair_plan["source_pair_plan_identity"],
        "ontology_identity": llm_arm["ontology_identity"],
        "quality_profile": EXPECTED_QUALITY_PROFILE,
        "quality_threshold_identity": bank_lock["quality_threshold_identity"],
        "quality_gate_thresholds": gate_profiles["profiles"][EXPECTED_QUALITY_PROFILE]["thresholds"],
        "q_used_for_selection": bank_lock["q_used_for_selection"],
        "by_route_quota": bank_lock["by_route"],
        "final_bank_size": bank_lock["final_bank_size"],
        "selected_set_identity": llm_arm["selected_set_identity"],
        "c3_bank_contract_identity": c3_bank["c3_bank_contract_identity"],
    }
    for name, expected in (
        ("ontology_identity", EXPECTED_ONTOLOGY_IDENTITY),
        ("gpat_checkpoint_sha256", EXPECTED_GPAT_CHECKPOINT_SHA256),
        ("physics_engine_version", EXPECTED_PHYSICS_ENGINE_VERSION),
        ("source_pair_plan_identity", EXPECTED_SOURCE_PAIR_PLAN_IDENTITY),
        ("package_identity", EXPECTED_PACKAGE_IDENTITY),
        ("renders_per_recipe", EXPECTED_RENDERS_PER_RECIPE),
        ("candidates_per_arm", EXPECTED_CANDIDATES_PER_ARM),
        ("quality_threshold_identity", EXPECTED_QUALITY_THRESHOLD_IDENTITY),
        ("final_bank_size", EXPECTED_FINAL_BANK_SIZE),
        ("selected_set_identity", EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY),
    ):
        if values[name] != expected:
            raise E6RenderError(
                f"historical value {name}={values[name]!r} disagrees with the pinned expectation "
                f"{expected!r}; the historical C5/C6 artifacts have drifted since this audit was written")
    if values["by_route_quota"] != EXPECTED_BY_ROUTE_QUOTA:
        raise E6RenderError("historical by_route quota has drifted from the pinned expectation")
    return values


# --------------------------------------------------------------------------- #
# C. Consume the frozen shuffle -- never regenerate, never mutate
# --------------------------------------------------------------------------- #

def load_frozen_shuffle_recipes(repo: Path) -> dict[str, Any]:
    """Reads `LLM_SHUFFLE_A_RECIPES.jsonl` verbatim. Never calls the shuffle
    generator (`c_ext_llm_shuffle.run_shuffle`/`main`), never rewrites it."""
    path = repo / training_plan.E6_SHUFFLE_RECIPES_PATH
    if not path.is_file():
        raise E6RenderError(f"missing frozen LLM-SHUFFLE-A recipes at {path.as_posix()}")
    raw_bytes = path.read_bytes()
    input_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lines = [line for line in raw_bytes.decode("utf-8").strip().split("\n") if line.strip()]
    recipes = [json.loads(line) for line in lines]
    content_identity = cc.sha256_json(recipes)
    return {"recipes": recipes, "content_identity": content_identity, "input_file_sha256": input_file_sha256}


def verify_shuffle_recipe_source(repo: Path) -> dict[str, Any]:
    """Fails closed unless the E6 training plan lock is FROZEN and this
    milestone's pinned LLM-SHUFFLE-A identity matches both the training-plan
    lock AND the recipes read fresh off disk right now."""
    plan_lock_path = repo / training_plan.TRAINING_PLAN_LOCK_PATH
    if not plan_lock_path.is_file():
        raise E6RenderError(f"missing frozen E6 training-plan lock at {plan_lock_path.as_posix()}")
    plan_lock = json.loads(plan_lock_path.read_text(encoding="utf-8"))
    if not training_plan.is_usable_plan_lock(plan_lock):
        raise E6RenderError("E6 training-plan lock is not usable (tampered, incomplete, or not FROZEN)")

    loaded = load_frozen_shuffle_recipes(repo)
    if loaded["content_identity"] != plan_lock["llm_shuffle_a_recipe_identity"]:
        raise E6RenderError(
            f"recipes.jsonl content identity {loaded['content_identity']!r} != the E6 training-plan "
            f"lock's llm_shuffle_a_recipe_identity {plan_lock['llm_shuffle_a_recipe_identity']!r}")
    if len(loaded["recipes"]) != plan_lock["recipe_count"]:
        raise E6RenderError(
            f"recipes.jsonl has {len(loaded['recipes'])} recipes, expected {plan_lock['recipe_count']}")
    return {"plan_lock": plan_lock, **loaded}


# --------------------------------------------------------------------------- #
# F. Source-pair / recipe alignment audit (mandatory before real render)
# --------------------------------------------------------------------------- #

def verify_source_pair_recipe_alignment(repo: Path, *, original_recipes: list[dict[str, Any]],
                                        shuffled_recipes: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK F: the base schedule (`c5_source_pair_plan.build_source_pair_plan`)
    binds each of its 2048 positions to a `recipe_ordinal` (an ARRAY INDEX,
    0..255) -- never to a recipe_id string -- and `c5_arm_plan.build_arm_plan`
    looks up `bank["recipes"][ordinal]` to resolve that position's actual
    recipe. `_recipe_id(recipe, ordinal)` (the identifier that becomes each
    candidate's `recipe_id` field) is therefore READ, not chosen, from
    whatever recipe object currently sits at that ordinal.

    `c_ext_llm_shuffle`'s six field-groups (medium, geometry, illumination,
    region, artifact_family_and_parameters, severity_group) never include a
    recipe's own `recipe_id`/`id`/`recipe_hash` field, and swaps preserve
    list LENGTH and ORDER (only field VALUES move between two ordinals) --
    so ordinal r's `_recipe_id` is IDENTICAL before and after the shuffle,
    even though ordinal r's field CONTENT has changed. This function proves
    that empirically, for every one of the 256 ordinals, rather than trusting
    the reasoning alone -- and fails closed if even one ordinal disagrees,
    since that would mean the shuffle altered recipe identity assignment
    itself (a different, unintended experiment).
    """
    from prism_fas.synthesis.c5_arm_plan import _recipe_id

    if len(original_recipes) != len(shuffled_recipes):
        raise E6RenderError(
            f"original ({len(original_recipes)}) and shuffled ({len(shuffled_recipes)}) recipe "
            "counts disagree; cannot verify ordinal alignment")
    mapping = []
    mismatches = []
    for ordinal, (original, shuffled) in enumerate(zip(original_recipes, shuffled_recipes)):
        original_id = _recipe_id(original, ordinal)
        shuffled_id = _recipe_id(shuffled, ordinal)
        mapping.append({"ordinal": ordinal, "original_recipe_id": original_id, "shuffled_recipe_id": shuffled_id})
        if original_id != shuffled_id:
            mismatches.append(ordinal)
    if mismatches:
        raise E6RenderError(
            f"source-pair/recipe alignment broken at {len(mismatches)} ordinal(s) "
            f"(first few: {mismatches[:5]}); the shuffle changed recipe identity "
            "assignment itself, not merely field content -- refusing to render "
            "against a different, unintended source-pair distribution")
    return {"schema_version": "e6-source-pair-alignment-v1", "ordinals_checked": len(mapping),
           "all_ordinals_aligned": True, "pairing_key": "recipe_ordinal (array index)",
           "mapping": mapping}


def write_source_pair_alignment_lock(repo: Path, alignment: dict[str, Any]) -> dict[str, Any]:
    body = {"schema_version": "e6-source-pair-alignment-lock-v1",
           "ordinals_checked": alignment["ordinals_checked"],
           "all_ordinals_aligned": alignment["all_ordinals_aligned"],
           "pairing_key": alignment["pairing_key"],
           "mapping_identity": cc.sha256_json(alignment["mapping"]), "status": "FROZEN"}
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


# --------------------------------------------------------------------------- #
# E. Render plan (own arm-plan identity; ARMS vocabulary stays closed)
# --------------------------------------------------------------------------- #

def _e6_arm_plan_identity(*, source_pair_plan_identity: str, recipe_bank_identity: str,
                          gpat_checkpoint_sha256: str, physics_engine_version: str,
                          ontology_identity: str, training_plan_identity: str) -> str:
    """Mirrors `c5_source_pair_plan.arm_candidate_plan_identity`'s SHAPE, but is
    a SEPARATE function: that frozen function's `arm in {RND,DET,LLM}` check
    is deliberately closed and must not be widened to admit a fourth arm."""
    material = {"schema_version": PLAN_SCHEMA_VERSION, "arm": E6_ARM_NAME,
               "source_pair_plan_identity": source_pair_plan_identity,
               "recipe_bank_identity": recipe_bank_identity,
               "gpat_checkpoint_sha256": gpat_checkpoint_sha256,
               "physics_engine_version": physics_engine_version,
               "ontology_identity": ontology_identity,
               "training_plan_identity": training_plan_identity}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def build_render_plan(repo: Path) -> dict[str, Any]:
    historical = audit_historical_path(repo)
    shuffle = verify_shuffle_recipe_source(repo)
    plan_lock = shuffle["plan_lock"]

    recipe_bank_identity = shuffle["content_identity"]
    arm_plan_identity = _e6_arm_plan_identity(
        source_pair_plan_identity=historical["source_pair_plan_identity"],
        recipe_bank_identity=recipe_bank_identity,
        gpat_checkpoint_sha256=historical["gpat_checkpoint_sha256"],
        physics_engine_version=historical["physics_engine_version"],
        ontology_identity=historical["ontology_identity"],
        training_plan_identity=plan_lock["plan_identity"])

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION, "milestone": "E6_RENDER", "arm": E6_ARM_NAME,
        "e6_training_plan_identity": plan_lock["plan_identity"],
        "original_llm_recipe_identity": historical["selected_set_identity"],
        "llm_shuffle_a_recipe_identity": recipe_bank_identity,
        "llm_shuffle_a_recipes_input_file_sha256": shuffle["input_file_sha256"],
        "shuffle_seed": training_plan.SHUFFLE_SEED, "recipe_count": len(shuffle["recipes"]),
        "arm_plan_identity": arm_plan_identity,
        "renderer": {"gpat_checkpoint_sha256": historical["gpat_checkpoint_sha256"],
                    "physics_engine_version": historical["physics_engine_version"]},
        "source_package_identity": historical["package_identity"],
        "source_pair_plan_identity": historical["source_pair_plan_identity"],
        "ontology_identity": historical["ontology_identity"],
        "candidates_per_arm": historical["candidates_per_arm"],
        "renders_per_recipe": historical["renders_per_recipe"],
        "quality": {"profile": historical["quality_profile"],
                   "threshold_identity": historical["quality_threshold_identity"],
                   "thresholds": historical["quality_gate_thresholds"],
                   "used_for_selection": historical["q_used_for_selection"]},
        "matching_policy": {"by_route_quota": historical["by_route_quota"],
                           "final_bank_size": historical["final_bank_size"]},
        "expected_candidate_count": historical["candidates_per_arm"],
        "expected_matched_bank_count": historical["final_bank_size"],
        "output_paths": {"candidates_root": CANDIDATES_ROOT, "matched_bank_root": MATCHED_BANK_ROOT,
                        "bank_lock": BANK_LOCK_PATH},
        "target_access": False, "llm_api_calls": 0,
    }
    plan["plan_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(plan))
    return plan


def build_parity_table(repo: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """TASK B: proves ORIGINAL_LLM and the E6 render plan agree on EVERY field
    except recipe identity/content. Fails closed on any other difference."""
    historical = audit_historical_path(repo)
    original = {
        "ontology_identity": historical["ontology_identity"],
        "gpat_checkpoint_sha256": historical["gpat_checkpoint_sha256"],
        "physics_engine_version": historical["physics_engine_version"],
        "source_pair_plan_identity": historical["source_pair_plan_identity"],
        "package_identity": historical["package_identity"],
        "renders_per_recipe": historical["renders_per_recipe"],
        "candidates_per_arm": historical["candidates_per_arm"],
        "quality_profile": historical["quality_profile"],
        "quality_threshold_identity": historical["quality_threshold_identity"],
        "final_bank_size": historical["final_bank_size"],
        "by_route_quota": historical["by_route_quota"],
    }
    e6_side = {
        "ontology_identity": plan["ontology_identity"],
        "gpat_checkpoint_sha256": plan["renderer"]["gpat_checkpoint_sha256"],
        "physics_engine_version": plan["renderer"]["physics_engine_version"],
        "source_pair_plan_identity": plan["source_pair_plan_identity"],
        "package_identity": plan["source_package_identity"],
        "renders_per_recipe": plan["renders_per_recipe"],
        "candidates_per_arm": plan["candidates_per_arm"],
        "quality_profile": plan["quality"]["profile"],
        "quality_threshold_identity": plan["quality"]["threshold_identity"],
        "final_bank_size": plan["matching_policy"]["final_bank_size"],
        "by_route_quota": plan["matching_policy"]["by_route_quota"],
    }
    rows = []
    mismatches = []
    for field in PARITY_FIELDS:
        matches = original[field] == e6_side[field]
        rows.append({"field": field, "original_llm": original[field], "llm_shuffle_a": e6_side[field],
                    "matches": matches})
        if not matches:
            mismatches.append(field)
    if mismatches:
        raise E6RenderError(f"render plan parity broken for field(s) other than recipe identity: {mismatches}")
    return {
        "schema_version": "e6-render-parity-audit-v1",
        "only_intended_difference": "recipe source identity/path, and consequently recipe content/joint structure",
        "recipe_identities": {"original_llm": plan["original_llm_recipe_identity"],
                             "llm_shuffle_a": plan["llm_shuffle_a_recipe_identity"]},
        "parity_rows": rows, "all_other_fields_match": True,
    }


def freeze_render_plan_lock(repo: Path, plan: dict[str, Any]) -> dict[str, Any]:
    existing_path = repo / RENDER_PLAN_LOCK_PATH
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("plan_identity") != plan["plan_identity"]:
            raise E6RenderError(
                "a DIFFERENT E6 render plan is already frozen; refusing to silently overwrite "
                f"(frozen={existing.get('plan_identity')!r}, recomputed={plan['plan_identity']!r})")
        return existing
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION, "plan_identity": plan["plan_identity"],
        "arm": plan["arm"], "e6_training_plan_identity": plan["e6_training_plan_identity"],
        "original_llm_recipe_identity": plan["original_llm_recipe_identity"],
        "llm_shuffle_a_recipe_identity": plan["llm_shuffle_a_recipe_identity"],
        "recipe_count": plan["recipe_count"], "arm_plan_identity": plan["arm_plan_identity"],
        "expected_candidate_count": plan["expected_candidate_count"],
        "expected_matched_bank_count": plan["expected_matched_bank_count"],
        "target_access": False, "llm_api_calls": 0, "status": "FROZEN",
    }
    lock["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(lock))
    return lock


def is_usable_render_plan_lock(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "FROZEN":
        return False
    body = {key: value for key, value in payload.items() if key != "lock_identity"}
    return cc.sha256_bytes(cc.canonical_json_bytes(body)) == payload.get("lock_identity")


# --------------------------------------------------------------------------- #
# C/K. GPU runtime resolution + preflight (no rendering, no heavy load)
# --------------------------------------------------------------------------- #

def resolve_gpu_runtime(repo: Path) -> dict[str, Any]:
    """Resolves every runtime input real C5 rendering needs, WITHOUT
    instantiating a route (`GPATRoute` loads the checkpoint into memory and
    requires CUDA) or opening the source image store's parquet manifest more
    than once. Reuses the canonical, already-established resolvers verbatim:

    * `pipeline.adapters.c4.verify_gpat_config_lock` -- the SAME function
      `pipeline.adapters.c5._verify_c4_lock` itself calls before rendering a
      single GPAT candidate through the checkpoint the lock names.
    * `synthesis.c5_render.scientific_device` -- CUDA-or-refuse, never a
      silent CPU fallback.
    * `synthesis.m8_pipeline.SampleStore.open` -- reads
      `manifests/source_train.parquet`; wrapped so a missing manifest (this
      laptop) is reported as unresolvable rather than raising.

    Never raises on an unresolvable runtime input -- that is what
    `_RESOLVABLE` fields are for; a genuinely BROKEN lock/identity still
    raises via the functions reused above.
    """
    from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock

    lock_path = repo / C4_SCIENTIFIC_LOCK_PATH
    verification = verify_gpat_config_lock(repo, lock_path)
    payload = verification["payload"]
    checkpoint_path = verification.get("checkpoint")
    checkpoint_sha256 = verification.get("checkpoint_sha256")
    checkpoint_exists = bool(checkpoint_path) and Path(checkpoint_path).is_file()
    measured_checkpoint_sha256 = verification.get("measured_checkpoint_sha256")

    manifest_path = repo / SOURCE_PACKAGE_ROOT / SOURCE_TRAIN_MANIFEST_RELATIVE
    manifest_exists = manifest_path.is_file()
    source_train_row_count = None
    source_store_resolvable = False
    if manifest_exists:
        try:
            import pyarrow.parquet as pq

            source_train_row_count = pq.read_metadata(manifest_path).num_rows
            source_store_resolvable = True
        except Exception:  # noqa: BLE001 - preflight must never crash on a bad manifest
            source_store_resolvable = False

    try:
        from prism_fas.synthesis.c5_render import ScientificDeviceUnavailable, scientific_device

        resolved_device = scientific_device()
        cuda_available = True
    except ScientificDeviceUnavailable:
        resolved_device, cuda_available = None, False
    except Exception:  # noqa: BLE001 - preflight must never crash on device probing
        resolved_device, cuda_available = None, False

    package_lock_path = repo / SOURCE_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    source_package_identity = None
    if package_lock_path.is_file():
        source_package_identity = json.loads(
            package_lock_path.read_text(encoding="utf-8")).get("content_identity_sha256")

    # ROUTES: `route_bank`/`build_routes` construct the two frozen routes
    # (physics, gpat) from `ROUTE_BY_SLOT` (a plain 2-element constant tuple)
    # -- reported here as a count + a deterministic digest over the two
    # identities that actually determine route BEHAVIOR, without
    # constructing a `GPATRoute` (which needs CUDA and loads the checkpoint).
    routes_count = 2
    routes_identity = cc.sha256_bytes(cc.canonical_json_bytes(
        {"physics_engine_version": EXPECTED_PHYSICS_ENGINE_VERSION,
         "gpat_checkpoint_sha256": EXPECTED_GPAT_CHECKPOINT_SHA256}))

    ontology_config_path = repo / ONTOLOGY_CONFIG_PATH
    quality_config_resolvable = (repo / C6_GATE_PROFILES_PATH).is_file()
    c6_matching_config_resolvable = (repo / C6_BANK_LOCK_LLM_PATH).is_file()
    quality_backend_assets = resolve_quality_backend_assets(repo)

    return {
        **quality_backend_assets,
        "schema_version": "e6-gpu-runtime-resolution-v1",
        "SOURCE_TRAIN_MANIFEST_PATH": manifest_path.relative_to(repo).as_posix(),
        "SOURCE_TRAIN_MANIFEST_EXISTS": manifest_exists,
        "SOURCE_TRAIN_ROW_COUNT": source_train_row_count,
        "SOURCE_PACKAGE_ROOT": SOURCE_PACKAGE_ROOT, "SOURCE_PACKAGE_IDENTITY": source_package_identity,
        "SOURCE_STORE_TYPE": "prism_fas.synthesis.m8_pipeline.SampleStore",
        "SOURCE_STORE_RESOLVABLE": source_store_resolvable,
        "ROUTES_COUNT": routes_count, "ROUTES_IDENTITY": routes_identity,
        "GPAT_CHECKPOINT_PATH": (Path(checkpoint_path).relative_to(repo).as_posix()
                                if checkpoint_path else None),
        "GPAT_CHECKPOINT_EXISTS": checkpoint_exists,
        "GPAT_CHECKPOINT_SHA256": checkpoint_sha256,
        "GPAT_CHECKPOINT_MEASURED_SHA256": measured_checkpoint_sha256,
        "GPAT_CHECKPOINT_RESOLVABLE": bool(checkpoint_sha256),
        "REQUESTED_DEVICE": "cuda", "RESOLVED_DEVICE": resolved_device, "CUDA_AVAILABLE": cuda_available,
        "CANDIDATES_ROOT": CANDIDATES_ROOT, "MATCHED_BANK_ROOT": MATCHED_BANK_ROOT,
        "ONTOLOGY_CONFIG_RESOLVABLE": ontology_config_path.is_file(),
        "QUALITY_CONFIG_RESOLVABLE": quality_config_resolvable,
        "C6_MATCHING_CONFIG_RESOLVABLE": c6_matching_config_resolvable,
        "c4_lock_ok": verification["ok"],
        "DEVICE_POLICY": "CUDA required for the GPAT route; the physics route is CPU-bound but is "
                         "always rendered through the SAME render_arm pass -- refuses rather than "
                         "silently falling back",
    }


def historical_quality_runtime_trace(repo: Path) -> dict[str, Any]:
    """TASK A: what runtime ACTUALLY produced the frozen ORIGINAL_LLM q values.

    The ONE frozen artifact that recorded it is `QUALITY_CALIBRATION.json`'s own
    `quality_backend_run_provenance` block, written by the SAME real C6
    `FIT_NOMINAL_CALIBRATION` run that fit the thresholds every candidate was
    later gated under -- not a separate log, not a guess. Cross-checked against
    two independent pieces of evidence:

    * the requirements tree (`requirements/{cpu,cuda-cu126,cuda-cu129,cuda-cu130}.txt`,
      `requirements/constraints.txt`) pins `onnxruntime==1.24.1` in EVERY profile,
      including the CUDA ones, and `onnxruntime-gpu` never appears anywhere in
      this repository -- `configs/environment/environment_contract.yaml`
      documents this as a deliberate cross-platform wheel-availability choice,
      not an oversight.
    * `quality_calibration.QualityBackends.__init__`'s own construction code:
      it REQUESTS `"CUDAExecutionProvider" if device.startswith("cuda") else
      "CPUExecutionProvider"`, then silently falls back to
      `"CPUExecutionProvider"` on ANY construction exception -- which
      requesting a provider absent from `onnxruntime.get_available_providers()`
      triggers.

    No artifact directly logs "provider X actually executed session.run" --
    that is DERIVED from the above two facts, never asserted as directly
    observed. Never infers CUDA merely because `requested_device` says "cuda":
    that field is what was ASKED for (and is what actually ran AdaFace/
    FaceXFormer, both torch models, on CUDA); the ONNX-Runtime-backed SCRFD
    detector is a separate runtime whose actual provider is derived
    separately, honestly labeled as derived.
    """
    calibration_path = repo / QUALITY_CALIBRATION_PATH
    evidence = [QUALITY_CALIBRATION_PATH, "requirements/cuda-cu129.txt", "requirements/cuda-cu126.txt",
               "requirements/cuda-cu130.txt", "requirements/cpu.txt", "requirements/constraints.txt",
               "configs/environment/environment_contract.yaml",
               "src/prism_fas/synthesis/quality_calibration.py (QualityBackends.__init__)"]
    if not calibration_path.is_file():
        return {"schema_version": "e6-historical-quality-runtime-trace-v1",
               "HISTORICAL_ORT_PACKAGE": "UNKNOWN", "HISTORICAL_ORT_VERSION": "UNKNOWN",
               "HISTORICAL_ORT_PROVIDER_REQUESTED": "UNKNOWN", "HISTORICAL_ORT_PROVIDER_ACTUAL": "UNKNOWN",
               "HISTORICAL_ORT_AVAILABLE_PROVIDERS": [], "HISTORICAL_SCRFD_INPUT_SIZE": "UNKNOWN",
               "HISTORICAL_QUALITY_DEVICE": "UNKNOWN", "EVIDENCE_ARTIFACTS": [],
               "reason": f"missing {QUALITY_CALIBRATION_PATH}; no frozen runtime provenance to read"}

    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    provenance = payload.get("quality_backend_run_provenance") or {}
    ort_version = provenance.get("onnxruntime")
    ort_available_providers = list(provenance.get("onnxruntime_providers") or [])
    requested_device = provenance.get("requested_device") or payload.get("device")
    requested_provider = ("CUDAExecutionProvider" if str(requested_device).startswith("cuda")
                          else "CPUExecutionProvider")
    actual_provider = (requested_provider if requested_provider in ort_available_providers
                       else "CPUExecutionProvider")
    detector_input_size = ((payload.get("quality_models") or {}).get("models", {})
                           .get("detector", {}) or {}).get("input_size")

    return {
        "schema_version": "e6-historical-quality-runtime-trace-v1",
        # `onnxruntime-gpu` never appears in the requirements tree; the same
        # `onnxruntime` (CPU-wheel) package is pinned in every profile,
        # including the CUDA ones -- a documented, deliberate choice, not
        # inferred from the providers list alone (either package can report
        # CPU-only providers if the CUDA/cuDNN shared libraries fail to load).
        "HISTORICAL_ORT_PACKAGE": "onnxruntime",
        "HISTORICAL_ORT_VERSION": ort_version or "UNKNOWN",
        "HISTORICAL_ORT_PROVIDER_REQUESTED": requested_provider,
        "HISTORICAL_ORT_PROVIDER_ACTUAL": actual_provider,
        "HISTORICAL_ORT_PROVIDER_ACTUAL_IS_DERIVED": True,
        "HISTORICAL_ORT_AVAILABLE_PROVIDERS": ort_available_providers,
        "HISTORICAL_SCRFD_INPUT_SIZE": detector_input_size,
        "HISTORICAL_QUALITY_DEVICE": requested_device,
        "HISTORICAL_QUALITY_DEVICE_NOTE": "the torch device for AdaFace/FaceXFormer (both ran on CUDA; "
                                          "cuda_available=True in the same provenance record). The SCRFD "
                                          "detector is a SEPARATE ONNX Runtime session whose actual "
                                          "provider is derived above, not this field.",
        "EVIDENCE_ARTIFACTS": evidence,
    }


def resolve_quality_backend_assets(repo: Path) -> dict[str, Any]:
    """TASK F: resolves every asset `default_metrics_provider` needs, WITHOUT
    loading a single model into memory (`resolve_weight` only stats the file
    and verifies its pinned SHA-256; it never opens the weight into a model).

    Mirrors `quality_calibration.QualityBackends.__init__`'s own three pinned
    roles (`identity`=AdaFace, `parsing`=FaceXFormer, `detector`=SCRFD, which
    ALSO supplies the landmark backend -- `QualityBackends.detect` returns both
    the detection score and the 5-point landmarks from the same SCRFD call) plus
    the fingerprint backend, which `synthesis.fingerprint` computes from pure
    NumPy and therefore has no model weight to resolve -- its only real
    dependency is the frozen `fingerprint.references` the calibration artifact
    itself carries.
    """
    from prism_fas.synthesis.quality_models import PINNED, QualityModelError, resolve_weight

    weight_root = repo / QUALITY_WEIGHT_ROOT

    def _resolve(role: str) -> tuple[str | None, str, bool]:
        spec = PINNED[role]
        try:
            path = resolve_weight(weight_root, role, verify=True)
        except QualityModelError:
            return None, str(spec["sha256"]), False
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:
            relative = path.as_posix()
        return relative, str(spec["sha256"]), True

    adaface_path, adaface_sha, adaface_weight_ok = _resolve("identity")
    parsing_path, parsing_sha, parsing_weight_ok = _resolve("parsing")
    detector_path, detector_sha, detector_weight_ok = _resolve("detector")

    # `DifferentiableAdaFace`/`FaceXFormerBackend` also need their vendored
    # third-party code trees, not only the weight file (`QualityModelRegistry
    # .code_root = weight_root / "code"`); resolving the weight alone would
    # under-report what `QualityBackends()` actually requires to construct.
    adaface_code_ok = (weight_root / "code" / "adaface" / "adaface_net.py").is_file()
    parsing_code_ok = (weight_root / "code" / "facexformer").is_dir()

    try:
        from prism_fas.synthesis.c5_render import ScientificDeviceUnavailable, scientific_device

        resolved_device = scientific_device()
    except ScientificDeviceUnavailable:
        resolved_device = None
    except Exception:  # noqa: BLE001 - preflight must never crash on device probing
        resolved_device = None

    # TASK G bug fix: the requested provider is a function of the TORCH device
    # (`QualityBackends.__init__`'s own `"CUDAExecutionProvider" if
    # device.startswith("cuda") else "CPUExecutionProvider"`), but whether that
    # provider is actually USABLE is a property of the ONNX RUNTIME install --
    # a SEPARATE runtime from torch. The previous version treated "torch
    # resolved a CUDA device" as sufficient for `LANDMARK_RESOLVABLE = true`,
    # which is exactly the bug this milestone's audit caught: torch having CUDA
    # says nothing about whether onnxruntime's own CUDAExecutionProvider is
    # importable/available. `onnxruntime.get_available_providers()` is the ONLY
    # authoritative source for that.
    try:
        import onnxruntime as ort

        ort_available_providers = list(ort.get_available_providers())
        ort_version = str(ort.__version__)
    except Exception:  # noqa: BLE001 - preflight must never crash if onnxruntime is unimportable
        ort_available_providers = []
        ort_version = None

    landmark_requested_provider = "CUDAExecutionProvider" if resolved_device else "CPUExecutionProvider"
    landmark_provider_available = landmark_requested_provider in ort_available_providers
    # `QualityBackends.__init__` wraps `SCRFDDetector(..., provider)` in a
    # try/except that silently falls back to `CPUExecutionProvider` on ANY
    # construction failure -- which requesting an unavailable provider
    # triggers. This mirrors that EXACT fallback, never a re-implementation.
    landmark_actual_provider = (landmark_requested_provider if landmark_provider_available
                                else "CPUExecutionProvider")
    # The construction never truly fails as long as CPUExecutionProvider (ORT's
    # always-compiled-in default) is available -- which it always is.
    landmark_runtime_resolvable = detector_weight_ok and "CPUExecutionProvider" in ort_available_providers
    provider = landmark_actual_provider  # kept for backward-compatible key name

    calibration_path = repo / QUALITY_CALIBRATION_PATH
    calibration_exists = calibration_path.is_file()
    calibration_identity_matches = False
    fingerprint_resolvable = False
    if calibration_exists:
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration_identity_matches = (
                str(payload.get("threshold_sha256")) == EXPECTED_QUALITY_THRESHOLD_IDENTITY)
            fingerprint_resolvable = bool((payload.get("fingerprint") or {}).get("references"))
        except (OSError, ValueError):
            calibration_identity_matches = False

    adaface_resolvable = adaface_weight_ok and adaface_code_ok
    landmark_resolvable = landmark_runtime_resolvable
    parsing_resolvable = parsing_weight_ok and parsing_code_ok
    quality_backends_resolvable = (adaface_resolvable and landmark_resolvable
                                   and parsing_resolvable and calibration_exists
                                   and calibration_identity_matches)

    historical_runtime = historical_quality_runtime_trace(repo)
    historical_actual_provider = historical_runtime.get("HISTORICAL_ORT_PROVIDER_ACTUAL")
    if historical_actual_provider in (None, "UNKNOWN"):
        quality_runtime_parity: bool | str = "UNKNOWN"
    else:
        quality_runtime_parity = landmark_actual_provider == historical_actual_provider

    return {
        "schema_version": "e6-quality-backend-assets-v1",
        "QUALITY_BACKENDS_CLASS": "prism_fas.synthesis.quality_calibration.QualityBackends",
        "QUALITY_BACKENDS_RESOLVABLE": quality_backends_resolvable,
        "ADAFACE_MODEL_PATH": adaface_path, "ADAFACE_MODEL_SHA256": adaface_sha,
        "ADAFACE_RESOLVABLE": adaface_resolvable,
        "LANDMARK_MODEL_PATH": detector_path, "LANDMARK_PROVIDER": provider,
        "LANDMARK_RESOLVABLE": landmark_resolvable,
        # TASK G: the provider-validation fix -- requested vs AVAILABLE
        # (onnxruntime.get_available_providers(), not torch's CUDA visibility)
        # vs the actually-used provider, kept as separate, honest fields.
        "LANDMARK_MODEL_RESOLVABLE": detector_weight_ok,
        "LANDMARK_REQUESTED_PROVIDER": landmark_requested_provider,
        "LANDMARK_PROVIDER_AVAILABLE": landmark_provider_available,
        "LANDMARK_ACTUAL_PROVIDER": landmark_actual_provider,
        "LANDMARK_RUNTIME_RESOLVABLE": landmark_runtime_resolvable,
        "ONNXRUNTIME_VERSION": ort_version, "ONNXRUNTIME_AVAILABLE_PROVIDERS": ort_available_providers,
        "QUALITY_RUNTIME_PARITY": quality_runtime_parity,
        "PARSING_MODEL_PATH": parsing_path, "PARSING_MODEL_SHA256": parsing_sha,
        "PARSING_RESOLVABLE": parsing_resolvable,
        "FINGERPRINT_BACKEND": "prism_fas.synthesis.fingerprint (deterministic Haar/gradient "
                               "descriptor over the rendered image; no trainable weights)",
        "FINGERPRINT_RESOLVABLE": fingerprint_resolvable,
        "QUALITY_CALIBRATION_PATH": QUALITY_CALIBRATION_PATH,
        "QUALITY_CALIBRATION_EXISTS": calibration_exists,
        "QUALITY_CONFIG_IDENTITY": EXPECTED_QUALITY_THRESHOLD_IDENTITY,
        "QUALITY_CONFIG_IDENTITY_MATCHES": calibration_identity_matches,
        "QUALITY_SCORER_SYMBOL": "prism_fas.synthesis.quality_gate.evaluate",
        "C6_MATCHER_SYMBOL": "prism_fas.synthesis.c6_matched_bank.select_route_bank",
        "NOT_IMPLEMENTED_SEAMS_REMAINING": [],
    }


def candidate_plan_contract_status(repo: Path) -> dict[str, Any]:
    """TASK H: whether the FULL 2048-row candidate-id/ontology render-row
    contract can be built and validated on THIS host, right now, with ZERO
    rendering. Runs the exact real functions `render_candidates_to_staging`
    itself calls before ever invoking a renderer
    (`build_arm_plan_rows` -> `candidate_identity` + `_assert_arm_plan`;
    `verify_source_pair_recipe_alignment`; `build_e6_route_bank` -> `load_ontology`),
    never a re-implementation.

    Never fabricates PASS: the real 2048-position base schedule needs the real
    `source_train` manifest, which is GPU-host-only. When it is not resolvable
    here, every contract field reports `UNRESOLVABLE_ON_THIS_HOST` rather than
    guessing an outcome only the GPU host can actually produce.
    """
    ontology_identity = None
    ontology_resolvable = False
    try:
        from prism_fas.recipes.ontology import load_ontology

        ontology = load_ontology(repo / ONTOLOGY_CONFIG_PATH)
        ontology_identity = ontology.sha256
        ontology_resolvable = ontology_identity == EXPECTED_ONTOLOGY_IDENTITY
    except Exception:  # noqa: BLE001 - contract status must never crash on a bad/missing config
        ontology_resolvable = False

    runtime = resolve_gpu_runtime(repo)
    base = {"schema_version": "e6-candidate-plan-contract-status-v1",
           "ONTOLOGY_RUNTIME_RESOLVABLE": ontology_resolvable, "ONTOLOGY_IDENTITY": ontology_identity}
    if not runtime["SOURCE_STORE_RESOLVABLE"]:
        return {**base, "CANDIDATE_ID_CONTRACT": "UNRESOLVABLE_ON_THIS_HOST",
               "CANDIDATE_ID_COUNT": None, "CANDIDATE_ID_UNIQUE_COUNT": None,
               "SOURCE_PAIR_PARITY": "UNRESOLVABLE_ON_THIS_HOST",
               "RENDER_ROW_CONTRACT": "UNRESOLVABLE_ON_THIS_HOST",
               "reason": "the real source_train manifest is not present on this host "
                         "(SOURCE_STORE_RESOLVABLE=False); the real 2048-position base "
                         "schedule cannot be recomputed here"}

    try:
        plan = build_render_plan(repo)
        shuffle = verify_shuffle_recipe_source(repo)
        rows = build_arm_plan_rows(repo, plan, shuffle["recipes"])
        ids = [row["candidate_id"] for row in rows]
        candidate_ok = (len(rows) == EXPECTED_CANDIDATES_PER_ARM
                        and len(set(ids)) == EXPECTED_CANDIDATES_PER_ARM)
        original_recipes = cc.read_jsonl(repo / "assets/recipe_banks/c3/llm/recipes.jsonl")
        alignment = verify_source_pair_recipe_alignment(
            repo, original_recipes=original_recipes, shuffled_recipes=shuffle["recipes"])
        parity_ok = bool(alignment["all_ordinals_aligned"])
        bank = build_e6_route_bank(repo, shuffle["recipes"], bank_identity=plan["llm_shuffle_a_recipe_identity"])
        render_row_ok = candidate_ok and parity_ok and "ontology" in bank and ontology_resolvable
        return {**base, "CANDIDATE_ID_CONTRACT": "PASS" if candidate_ok else "FAIL",
               "CANDIDATE_ID_COUNT": len(rows), "CANDIDATE_ID_UNIQUE_COUNT": len(set(ids)),
               "SOURCE_PAIR_PARITY": "PASS" if parity_ok else "FAIL",
               "RENDER_ROW_CONTRACT": "PASS" if render_row_ok else "FAIL"}
    except E6RenderError as error:
        return {**base, "CANDIDATE_ID_CONTRACT": "FAIL", "CANDIDATE_ID_COUNT": None,
               "CANDIDATE_ID_UNIQUE_COUNT": None, "SOURCE_PAIR_PARITY": "FAIL",
               "RENDER_ROW_CONTRACT": "FAIL", "reason": str(error)}


def run_preflight(repo: Path) -> dict[str, Any]:
    plan = build_render_plan(repo)
    parity = build_parity_table(repo, plan)
    runtime = resolve_gpu_runtime(repo)
    candidate_contract = {key: value for key, value in candidate_plan_contract_status(repo).items()
                         if key != "schema_version"}
    return {
        "schema_version": "e6-render-preflight-v1",
        "RECIPE_IDENTITY": plan["llm_shuffle_a_recipe_identity"], "RECIPE_COUNT": plan["recipe_count"],
        "RENDERER_IDENTITY": plan["renderer"], "SOURCE_PACKAGE_IDENTITY": plan["source_package_identity"],
        "QUALITY_CONFIG_IDENTITY": plan["quality"]["threshold_identity"],
        "MATCHING_CONFIG_IDENTITY": cc.sha256_bytes(cc.canonical_json_bytes(plan["matching_policy"])),
        "EXPECTED_CANDIDATE_COUNT": plan["expected_candidate_count"],
        "EXPECTED_MATCHED_BANK_COUNT": plan["expected_matched_bank_count"],
        "GPU_AVAILABLE": runtime["CUDA_AVAILABLE"], "TARGET_ACCESS": False, "LLM_API_CALLS": 0,
        "candidate_rendered": False, "parity_check": parity["all_other_fields_match"],
        # TASK H: the second GPU preflight contract -- every locator a real
        # `--execute --authorize-gpu-render` needs, surfaced without rendering.
        "SOURCE_TRAIN_MANIFEST_PATH": runtime["SOURCE_TRAIN_MANIFEST_PATH"],
        "SOURCE_TRAIN_ROW_COUNT": runtime["SOURCE_TRAIN_ROW_COUNT"],
        "SOURCE_STORE_CLASS": runtime["SOURCE_STORE_TYPE"],
        "SOURCE_STORE_RESOLVABLE": runtime["SOURCE_STORE_RESOLVABLE"],
        "GPAT_ROUTE_CLASS": "prism_fas.synthesis.synthetic_bank.GPATRoute",
        "PHYSICS_ROUTE_CLASS": "prism_fas.synthesis.synthetic_bank.PhysicsRoute",
        "ROUTES_COUNT": runtime["ROUTES_COUNT"],
        "GPAT_CHECKPOINT_PATH": runtime["GPAT_CHECKPOINT_PATH"],
        "GPAT_CHECKPOINT_SHA256": runtime["GPAT_CHECKPOINT_SHA256"],
        "REQUESTED_DEVICE": runtime["REQUESTED_DEVICE"], "RESOLVED_DEVICE": runtime["RESOLVED_DEVICE"],
        "CUDA_AVAILABLE": runtime["CUDA_AVAILABLE"],
        "QUALITY_BACKENDS_CLASS": runtime["QUALITY_BACKENDS_CLASS"],
        "QUALITY_BACKENDS_RESOLVABLE": runtime["QUALITY_BACKENDS_RESOLVABLE"],
        "ADAFACE_MODEL_PATH": runtime["ADAFACE_MODEL_PATH"],
        "ADAFACE_MODEL_SHA256": runtime["ADAFACE_MODEL_SHA256"],
        "ADAFACE_RESOLVABLE": runtime["ADAFACE_RESOLVABLE"],
        "LANDMARK_MODEL_PATH": runtime["LANDMARK_MODEL_PATH"],
        "LANDMARK_PROVIDER": runtime["LANDMARK_PROVIDER"],
        "LANDMARK_RESOLVABLE": runtime["LANDMARK_RESOLVABLE"],
        "LANDMARK_MODEL_RESOLVABLE": runtime["LANDMARK_MODEL_RESOLVABLE"],
        "LANDMARK_REQUESTED_PROVIDER": runtime["LANDMARK_REQUESTED_PROVIDER"],
        "LANDMARK_PROVIDER_AVAILABLE": runtime["LANDMARK_PROVIDER_AVAILABLE"],
        "LANDMARK_ACTUAL_PROVIDER": runtime["LANDMARK_ACTUAL_PROVIDER"],
        "LANDMARK_RUNTIME_RESOLVABLE": runtime["LANDMARK_RUNTIME_RESOLVABLE"],
        "QUALITY_RUNTIME_PARITY": runtime["QUALITY_RUNTIME_PARITY"],
        "ONNXRUNTIME_VERSION": runtime["ONNXRUNTIME_VERSION"],
        "ONNXRUNTIME_AVAILABLE_PROVIDERS": runtime["ONNXRUNTIME_AVAILABLE_PROVIDERS"],
        "PARSING_MODEL_PATH": runtime["PARSING_MODEL_PATH"],
        "PARSING_MODEL_SHA256": runtime["PARSING_MODEL_SHA256"],
        "PARSING_RESOLVABLE": runtime["PARSING_RESOLVABLE"],
        "FINGERPRINT_BACKEND": runtime["FINGERPRINT_BACKEND"],
        "FINGERPRINT_RESOLVABLE": runtime["FINGERPRINT_RESOLVABLE"],
        "QUALITY_SCORER_SYMBOL": runtime["QUALITY_SCORER_SYMBOL"],
        "C6_MATCHER_SYMBOL": runtime["C6_MATCHER_SYMBOL"],
        "QUALITY_DEPENDENCIES_RESOLVABLE": runtime["QUALITY_BACKENDS_RESOLVABLE"],
        **candidate_contract,
        "NOT_IMPLEMENTED_SEAMS_REMAINING": [],
        "RENDER_EXECUTED": False,
        "runtime": runtime,
    }


# --------------------------------------------------------------------------- #
# F/G. Candidate rendering + resume-safe promotion (GPU-only; injectable)
# --------------------------------------------------------------------------- #

def build_arm_plan_rows(repo: Path, plan: dict[str, Any], recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Binds the FROZEN, arm-independent base schedule positions
    (`C5_SOURCE_PAIR_PLAN.json`, read verbatim, never rebuilt) to
    LLM-SHUFFLE-A's shuffled recipes -- exactly mirroring
    `c5_arm_plan.build_arm_plan`'s own loop shape, without calling that
    closed-vocabulary function directly.

    The frozen `C5_SOURCE_PAIR_PLAN.json` report serializes only the POSITION
    COUNT, not the 2048-row array itself (`positions` is excluded from that
    lock's own identity material and persisted as a plain int) -- the real
    array must be RECOMPUTED via the same deterministic
    `synthesis.c5_source_pair_plan.build_source_pair_plan(package_root, seed=...)`
    historical C5 used, then verified to reproduce the pinned
    `source_pair_plan_identity` exactly (mirroring how this milestone already
    re-derives the LLM-SHUFFLE-A recipe shuffle rather than trusting a
    snapshot). This needs the real `source_train` manifest, which is GPU-only
    on this laptop -- fails closed here otherwise, never fabricates a
    positions array.
    """
    from prism_fas.synthesis.c5_arm_plan import _assert_arm_plan, _recipe_id
    from prism_fas.synthesis.c5_source_pair_plan import (PLAN_SEED, build_source_pair_plan,
                                                         candidate_identity,
                                                         source_pair_plan_identity)

    package_root = repo / "data/packages/prism_data_v1_m3b"
    manifest_path = package_root / "manifests/source_train.parquet"
    if not manifest_path.is_file():
        raise E6RenderError(
            f"missing {manifest_path.as_posix()}; the real source_train manifest is required to "
            "recompute the base candidate schedule and is GPU-only on this laptop -- refusing to "
            "fabricate the 2048-position schedule from the frozen summary alone")
    base_plan = build_source_pair_plan(package_root, seed=PLAN_SEED)
    recomputed_identity = source_pair_plan_identity(base_plan)
    if recomputed_identity != plan["source_pair_plan_identity"]:
        raise E6RenderError(
            f"recomputed source_pair_plan_identity {recomputed_identity!r} != the pinned "
            f"{plan['source_pair_plan_identity']!r}; the historical base schedule has drifted "
            "or could not be reproduced")
    positions = base_plan["positions"]
    rows = []
    for row in positions:
        ordinal = int(row["recipe_ordinal"])
        recipe_id = _recipe_id(recipes[ordinal], ordinal)
        binding = (plan["renderer"]["gpat_checkpoint_sha256"] if row["route"] == "gpat"
                  else plan["renderer"]["physics_engine_version"])
        # TASK B: the SAME historical per-candidate identity function
        # `c5_arm_plan.build_arm_plan` calls -- `candidate_identity` carries no
        # closed-arm-vocabulary check (see E6_CANDIDATE_ID_TRACE.json), so it is
        # reused here verbatim, unmodified, with `arm=E6_ARM_NAME`. No new
        # E6-specific candidate-id convention is introduced.
        candidate_id = candidate_identity(
            source_pair_plan_identity=plan["source_pair_plan_identity"], arm=E6_ARM_NAME,
            recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=recipe_id,
            recipe_ordinal=ordinal, slot=int(row["slot"]), position=int(row["position"]),
            route=row["route"], live_target_sample_id=row["live_target_sample_id"],
            spoof_source_sample_id=row["spoof_source_sample_id"],
            package_identity=plan["source_package_identity"], ontology_identity=plan["ontology_identity"],
            generator_binding=binding)
        rows.append({**row, "arm": E6_ARM_NAME, "recipe_id": recipe_id, "candidate_id": candidate_id,
                    "recipe_bank_identity": plan["llm_shuffle_a_recipe_identity"],
                    "generator_binding": binding})
    # TASK B: the SAME historical cardinality/uniqueness/route-split/per-recipe
    # assertion `c5_arm_plan.build_arm_plan` runs on its own 2048 rows -- fails
    # closed (ArmPlanError) on any duplicate id, wrong count, or uneven split.
    _assert_arm_plan(rows, E6_ARM_NAME)
    return rows


def build_e6_route_bank(repo: Path, recipes: list[dict[str, Any]], *, bank_identity: str) -> dict[str, Any]:
    """Mirrors `synthesis.c5_render.route_bank`'s SHAPE exactly (the dict
    `PhysicsRoute`/`GPATRoute`/`render_one` consume), built from LLM-SHUFFLE-A's
    frozen recipes instead of calling `route_bank` (which is closed to
    `{RND, DET, LLM}` via `c5_arm_plan.arm_bank_root`).

    TASK D: carries the REAL, live `ontology` object (`recipes.ontology.Ontology`,
    from the SAME `load_ontology` call historical `route_bank` makes -- see
    `E6_ONTOLOGY_RUNTIME_TRACE.json`), not only its identity string.
    `synthetic_bank.PhysicsRoute.generate`/`GPATRoute.generate` read
    `bank["ontology"]` directly to call `compile_recipe(recipe, bank["ontology"],
    bank_id=bank["bank_id"])`; a bank carrying only `ontology_identity` raises
    `KeyError("ontology")` there. No historical bank ever persists the Ontology
    object itself -- it is always re-loaded fresh from
    `configs/recipes/ontology_m7.yaml` and identity-checked, exactly as here.
    """
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    ontology = load_ontology(repo / ONTOLOGY_CONFIG_PATH)
    if ontology.sha256 != EXPECTED_ONTOLOGY_IDENTITY:
        raise E6RenderError(
            f"resolved ontology identity {ontology.sha256!r} != pinned {EXPECTED_ONTOLOGY_IDENTITY!r}")
    parsed = [parse_recipe(recipe) for recipe in recipes]
    return {"recipes": parsed, "bank_id": f"e6_{E6_ARM_NAME.lower()}", "bank_identity": bank_identity,
           "ontology": ontology, "ontology_identity": ontology.sha256}


def resolve_render_runtime_objects(repo: Path) -> dict[str, Any]:
    """The REAL, GPU-only construction of `store` + `routes`: reuses
    `synthesis.m8_pipeline.SampleStore.open`, `pipeline.adapters.c4.verify_gpat_config_lock`
    and `synthesis.c5_render.build_routes` unmodified. Raises with a clear
    reason if any input is unresolvable (missing source_train manifest, no
    CUDA, missing checkpoint bytes) -- never fabricates a placeholder."""
    from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock
    from prism_fas.synthesis import c5_render
    from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit

    device = c5_render.scientific_device()  # raises ScientificDeviceUnavailable without CUDA
    verification = verify_gpat_config_lock(repo, repo / C4_SCIENTIFIC_LOCK_PATH)
    if not verification["ok"]:
        raise E6RenderError("the frozen C4 GPAT lock does not verify; refusing to build routes")
    store = SampleStore.open(repo / SOURCE_PACKAGE_ROOT, SourceOnlyAudit())
    payload = verification["payload"]
    expected_identity = {key: payload.get(key) for key in
                        ("package_identity", "recipe_bank_identity", "pair_plan_identity",
                         "config_hash", "architecture_hash", "adaface_weight_sha256")}
    routes = c5_render.build_routes(repo, checkpoint_path=verification["checkpoint"],
                                    checkpoint_sha256=str(verification["checkpoint_sha256"]),
                                    expected_identity=expected_identity, device=device)
    return {"store": store, "routes": routes, "device": device}


def identity_for_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """A third pre-render contract gap, found while proving Task G's dry
    execution boundary: `c5_render.identity_for(row, plan)` reads
    `plan["package_identity"]` and `plan["recipe_bank_identity"]`
    (`c5_arm_plan.build_arm_plan`'s own arm-plan field names), but this
    module's `build_render_plan` names the same two identities
    `source_package_identity` and `llm_shuffle_a_recipe_identity` -- clearer
    names for an E6 plan that must also carry the ORIGINAL_LLM recipe identity
    alongside the shuffled one. Renaming `build_render_plan`'s own fields would
    change `plan_identity`'s hash material and every other reader of this
    dict; instead this is a small, additive, read-only alias view, built fresh
    at the one real call site that needs the historical names.
    """
    return {**plan, "package_identity": plan["source_package_identity"],
           "recipe_bank_identity": plan["llm_shuffle_a_recipe_identity"]}


def default_candidate_renderer(*, repo: Path, plan: dict[str, Any], row: dict[str, Any],
                               candidates_root: Path, bank: dict[str, Any] | None = None,
                               store: Any = None, routes: dict[str, Any] | None = None
                               ) -> dict[str, Any]:
    """The REAL single-candidate render: reuses `synthesis.c5_render.identity_for`
    / `render_one` and `synthesis.c5_raw_generation.candidate_dir` /
    `reuse_decision` / `write_record` / `write_payload_bytes` unmodified,
    against a real `store`/`routes`/`bank` (resolved by
    `resolve_render_runtime_objects`/`build_e6_route_bank` -- GPU-only, never
    fabricated). If `store`/`routes` are not supplied, resolves them for
    real (raises on this laptop, exactly as `resolve_gpu_runtime` already
    reports). Every test injects a fake `candidate_renderer` instead of
    calling this function at all.
    """
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c5_render

    identity = c5_render.identity_for(row, identity_for_plan(plan))
    directory = raw.candidate_dir(candidates_root, E6_ARM_NAME, row["candidate_id"])
    directory.mkdir(parents=True, exist_ok=True)
    decision = raw.reuse_decision(directory, identity)
    if decision.get("reusable"):
        return decision

    if store is None or routes is None:
        runtime = resolve_render_runtime_objects(repo)
        store, routes = runtime["store"], runtime["routes"]
    if bank is None:
        raise E6RenderError("default_candidate_renderer requires a real `bank` dict (build_e6_route_bank)")

    route = routes[row["route"]]
    result, trace = c5_render.render_one(store, bank, route, row)
    payloads = raw.write_payload_bytes(directory, result)
    record = raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                 payload_sha256=payloads, trace=trace)
    raw.write_record(directory, record)
    return {"reusable": True, "reason": "GENERATED", "candidate_id": identity.candidate_id,
           "payload_sha256": payloads}


def render_candidates_to_staging(*, repo: Path, plan: dict[str, Any], recipes: list[dict[str, Any]],
                                 candidate_renderer: Callable[..., dict[str, Any]] | None = None
                                 ) -> dict[str, Any]:
    """Never publishes anything outside `CANDIDATES_ROOT`; candidate-level
    resume safety is entirely the REUSED `reuse_decision`/`write_record`
    machinery (a candidate record is written LAST, atomically -- its
    presence is what makes a resumed render skip it). `store`/`routes`/`bank`
    are resolved ONCE here (never per-candidate) when the real renderer is
    in use; an injected fake renderer never needs them."""
    renderer = candidate_renderer or default_candidate_renderer
    rows = build_arm_plan_rows(repo, plan, recipes)
    candidates_root = repo / CANDIDATES_ROOT
    shared: dict[str, Any] = {}
    if renderer is default_candidate_renderer:
        runtime = resolve_render_runtime_objects(repo)
        shared = {"store": runtime["store"], "routes": runtime["routes"],
                 "bank": build_e6_route_bank(repo, recipes, bank_identity=plan["llm_shuffle_a_recipe_identity"])}
    results = [renderer(repo=repo, plan=plan, row=row, candidates_root=candidates_root, **shared)
              for row in rows]
    return {"rows": rows, "results": results, "candidates_root": str(candidates_root)}


# --------------------------------------------------------------------------- #
# H. Quality (q) post-render audit -- never fabricated, never q-matched
# --------------------------------------------------------------------------- #

def compute_q_summary(q_values: list[float]) -> dict[str, Any]:
    """n / mean / sample SD / median / Q1 / Q3, the SAME representation
    historical LLM's realized q was measured with."""
    import statistics

    if not q_values:
        raise E6RenderError("cannot summarize q over zero candidates")
    values = sorted(float(v) for v in q_values)
    n = len(values)
    return {"n": n, "mean": cc.mean(values), "sample_sd": cc.sample_sd(values),
           "median": statistics.median(values),
           "q1": statistics.quantiles(values, n=4)[0] if n >= 4 else values[0],
           "q3": statistics.quantiles(values, n=4)[2] if n >= 4 else values[-1]}


def standardized_mean_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    """SMD = (mean_a - mean_b) / pooled_sd. Symmetric convention: ORIGINAL_LLM
    is `a`, LLM_SHUFFLE_A is `b` -- a negative SMD means LLM_SHUFFLE_A's mean
    q is HIGHER than ORIGINAL_LLM's."""
    import math

    pooled_sd = math.sqrt((a["sample_sd"] ** 2 + b["sample_sd"] ** 2) / 2.0)
    if pooled_sd == 0.0:
        return 0.0
    return (a["mean"] - b["mean"]) / pooled_sd


def e8_trigger(smd: float) -> bool:
    return abs(smd) >= E8_SMD_TRIGGER_THRESHOLD


def build_q_audit(*, original_llm_q: dict[str, Any], shuffle_a_q_values: list[float]) -> dict[str, Any]:
    shuffle_a_q = compute_q_summary(shuffle_a_q_values)
    smd = standardized_mean_difference(original_llm_q, shuffle_a_q)
    return {"schema_version": "e6-q-audit-v1", "original_llm_q": original_llm_q,
           "llm_shuffle_a_q": shuffle_a_q, "smd_q": smd,
           "e8_smd_trigger_threshold": E8_SMD_TRIGGER_THRESHOLD, "e8_q_match_trigger": e8_trigger(smd),
           "q_matched": False, "quality_weights_altered": False, "e8_triggered_automatically": False}


# --------------------------------------------------------------------------- #
# I. Final matched-bank lock
# --------------------------------------------------------------------------- #

def build_matched_bank_lock(*, plan: dict[str, Any], selected: list[dict[str, Any]],
                            source_pair_alignment_lock: dict[str, Any] | None = None,
                            q_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mirrors the historical `C6_BANK_LOCK_LLM.json` shape closely enough for
    `C6MatchedBankReader.open` to accept it (that function needs only
    `selected`, `selected_set_sha256`, `quality_threshold_identity`), while
    binding every identity TASK I requires."""
    if len(selected) != plan["expected_matched_bank_count"]:
        raise E6RenderError(
            f"matched bank has {len(selected)} rows, expected {plan['expected_matched_bank_count']}")
    route_counts: dict[str, int] = {}
    for row in selected:
        route = str(row.get("route", ""))
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
    body = {
        "schema_version": "e6-shuffle-a-bank-lock-v1", "arm": E6_ARM_NAME,
        "e6_render_plan_identity": plan["plan_identity"],
        "e6_training_plan_identity": plan["e6_training_plan_identity"],
        "llm_shuffle_a_recipe_identity": plan["llm_shuffle_a_recipe_identity"],
        "llm_shuffle_a_recipes_input_file_sha256": plan["llm_shuffle_a_recipes_input_file_sha256"],
        "source_package_identity": plan["source_package_identity"],
        "source_pair_plan_identity": plan["source_pair_plan_identity"],
        "ontology_identity": plan["ontology_identity"],
        "gpat_checkpoint_sha256": plan["renderer"]["gpat_checkpoint_sha256"],
        "physics_engine_version": plan["renderer"]["physics_engine_version"],
        "source_pair_alignment_lock_identity": (source_pair_alignment_lock or {}).get("lock_identity"),
        "candidate_count": plan["expected_candidate_count"],
        "quality_threshold_identity": plan["quality"]["threshold_identity"],
        "quality_pass_statistics": {"selected": len(selected), "planned": plan["expected_candidate_count"]},
        "matching_config_identity": cc.sha256_bytes(cc.canonical_json_bytes(plan["matching_policy"])),
        "final_bank_size": len(selected), "route_counts": route_counts,
        "by_route": plan["matching_policy"]["by_route_quota"],
        "selected": selected,
        "q_audit_identity": cc.sha256_bytes(cc.canonical_json_bytes(q_audit)) if q_audit else None,
        "target_access": False, "llm_api_calls": 0, "status": "FROZEN",
    }
    body["selected_set_sha256"] = cc.sha256_json(selected)
    body["bank_lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return {**body, "lock_identity": body["bank_lock_identity"]}


def is_usable_bank_lock(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "FROZEN":
        return False
    body = {key: value for key, value in payload.items() if key not in ("bank_lock_identity", "lock_identity")}
    recomputed = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return recomputed == payload.get("bank_lock_identity") == payload.get("lock_identity")


def verify_bank_readable_by_c6_matched_bank_reader(*, candidates_root: Path, bank_lock: dict[str, Any],
                                                   recipes: list[dict[str, Any]]) -> Any:
    """TASK I: proves the produced bank is readable by the UNMODIFIED,
    historical `C6MatchedBankReader.open` -- no E6-specific dataset reader is
    added anywhere in this module."""
    from prism_fas.detector.c6_bank import C6MatchedBankReader

    return C6MatchedBankReader.open(
        candidates_root=candidates_root, arm=E6_ARM_NAME, bank_lock=bank_lock, recipes=recipes,
        package_identity=bank_lock["source_package_identity"],
        recipe_bank_identity=bank_lock["llm_shuffle_a_recipe_identity"],
        expected_selected_set_sha256=bank_lock["selected_set_sha256"])


def build_provenance(repo: Path, plan: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        commit = ""
    return {"schema_version": "e6-render-provenance-v1", "code_commit": commit,
           "plan_identity": plan["plan_identity"], "lock_identity": lock["lock_identity"],
           "target_features_accessed": False, "target_labels_accessed": False, "llm_api_calls": 0,
           "detector_training_executed": False, "image_rendering_executed": False}


def write_e6_render_preparation(repo: Path) -> dict[str, str]:
    plan = build_render_plan(repo)
    parity = build_parity_table(repo, plan)
    lock = freeze_render_plan_lock(repo, plan)
    provenance = build_provenance(repo, plan, lock)
    written = {
        "plan": cc.write_json_atomic(RENDER_PLAN_PATH, plan, root=repo),
        "parity_audit": cc.write_json_atomic(RENDER_PARITY_AUDIT_PATH, parity, root=repo),
        "provenance": cc.write_json_atomic(RENDER_PROVENANCE_PATH, provenance, root=repo),
        "plan_lock": cc.write_json_atomic(RENDER_PLAN_LOCK_PATH, lock, root=repo),
    }
    return written


# --------------------------------------------------------------------------- #
# D. Real execution wiring (GPU-only; every heavy step is an injectable seam)
# --------------------------------------------------------------------------- #

#: The resolved quality runtime (backends + calibration + evaluator + store +
#: bank), built ONCE per process and reused for every candidate -- never
#: rebuilt per candidate. `reset_quality_runtime_cache_for_tests` is the only
#: sanctioned way to clear it (tests only; real execution runs once per process
#: and never needs to).
_QUALITY_RUNTIME_CACHE: dict[str, Any] | None = None


def reset_quality_runtime_cache_for_tests() -> None:
    """Clears the module-level quality-runtime cache. Test-only seam: real
    execution builds the cache exactly once per process and never resets it."""
    global _QUALITY_RUNTIME_CACHE
    _QUALITY_RUNTIME_CACHE = None


def _resolve_quality_bank(repo: Path) -> dict[str, Any]:
    """The `bank` shape `synthesis.c6_scientific.requested_support_for` needs
    (`bank["recipes"]`, `bank["ontology"]`, `bank["bank_id"]`) -- built from
    LLM-SHUFFLE-A's own frozen, verified recipes, exactly like
    `build_e6_route_bank` resolves them for rendering. Kept as its OWN small
    helper (never calling or editing `build_e6_route_bank`, which the candidate
    renderer owns) so this quality-measurement seam does not reach into, or
    alter, the render wiring TASK E explicitly leaves untouched.
    """
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    shuffle = verify_shuffle_recipe_source(repo)
    ontology = load_ontology(repo / ONTOLOGY_CONFIG_PATH)
    if ontology.sha256 != EXPECTED_ONTOLOGY_IDENTITY:
        raise E6RenderError(
            f"resolved ontology identity {ontology.sha256!r} != pinned {EXPECTED_ONTOLOGY_IDENTITY!r}")
    parsed = [parse_recipe(recipe) for recipe in shuffle["recipes"]]
    return {"recipes": parsed, "bank_id": f"e6_{E6_ARM_NAME.lower()}",
           "bank_identity": shuffle["content_identity"], "ontology": ontology,
           "ontology_identity": ontology.sha256}


def _resolve_quality_model_runtime(repo: Path) -> dict[str, Any]:
    """Builds the REAL, arm-INDEPENDENT quality-model runtime exactly once
    per process: the frozen NOMINAL calibration
    (`quality_calibration.FrozenCalibration`, bound to
    `EXPECTED_QUALITY_THRESHOLD_IDENTITY` -- fails closed on drift, never
    refits), the three pinned model backends
    (`quality_calibration.QualityBackends`), the canonical per-candidate
    evaluator (`synthetic_bank.CandidateEvaluator`) that wraps them, and the
    source image store (`m8_pipeline.SampleStore`). GPU-only:
    `c5_render.scientific_device` raises `ScientificDeviceUnavailable`
    without CUDA, with no silent CPU fallback -- matching every other real
    runtime resolver in this module.

    Deliberately excludes the recipe BANK: which bank is correct to
    reconstruct a candidate's requested support/strength depends on which
    recipe content actually rendered that candidate (`_resolve_quality_bank`/
    `default_metrics_provider`'s `quality_bank` override), and is therefore
    never safe to fold into this arm-independent, once-per-process cache.
    """
    global _QUALITY_RUNTIME_CACHE
    if _QUALITY_RUNTIME_CACHE is not None:
        return _QUALITY_RUNTIME_CACHE

    from prism_fas.synthesis import c5_render
    from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit
    from prism_fas.synthesis.quality_calibration import QualityBackends
    from prism_fas.synthesis.synthetic_bank import CandidateEvaluator, FrozenCalibration

    calibration_path = repo / QUALITY_CALIBRATION_PATH
    if not calibration_path.is_file():
        raise E6RenderError(f"missing frozen quality calibration at {calibration_path.as_posix()}")
    calibration = FrozenCalibration.load(calibration_path)
    if calibration.threshold_sha256 != EXPECTED_QUALITY_THRESHOLD_IDENTITY:
        raise E6RenderError(
            f"quality calibration threshold identity {calibration.threshold_sha256!r} != the pinned "
            f"{EXPECTED_QUALITY_THRESHOLD_IDENTITY!r}; refusing to gate against a drifted calibration")

    device = c5_render.scientific_device()  # raises ScientificDeviceUnavailable without CUDA
    backends = QualityBackends(repo / QUALITY_WEIGHT_ROOT, device=device)
    evaluator = CandidateEvaluator(backends, calibration)
    store = SampleStore.open(repo / SOURCE_PACKAGE_ROOT, SourceOnlyAudit())

    _QUALITY_RUNTIME_CACHE = {"evaluator": evaluator, "store": store,
                              "calibration": calibration, "backends": backends, "device": device}
    return _QUALITY_RUNTIME_CACHE


def _resolve_quality_runtime(repo: Path, *, quality_bank: Mapping[str, Any] | None = None
                             ) -> dict[str, Any]:
    """The historical, byte-for-byte-preserved entry point: the arm-
    independent model runtime plus a `bank`. Omitting `quality_bank`
    resolves the EXACT historical default (`_resolve_quality_bank`, i.e. the
    LLM-SHUFFLE-A recipe bank) -- every historical E6 caller omits it and
    sees no behavior change. A caller that supplies `quality_bank` (E6-v2,
    per-arm) gets ITS bank used for reconstruction instead -- this was
    ATTEMPT-2's proven third bug: `requested_support_for`/
    `reconstruct_discrete` silently reconstructed every arm's candidates,
    including `LLM_ORIGINAL_CURRENT_V2`'s, against the SHUFFLE-A recipe
    CONTENT (same `recipe_id`, different fields per
    `_resolve_historical_llm_bank`'s docstring), corrupting the requested
    support mask/strength and collapsing quality-gate pass rates for any arm
    that was not LLM-SHUFFLE-A itself.
    """
    model = _resolve_quality_model_runtime(repo)
    bank = quality_bank if quality_bank is not None else _resolve_quality_bank(repo)
    return {**model, "bank": bank}


def default_metrics_provider(*, repo: Path, row: dict[str, Any], record: dict[str, Any],
                             candidates_root: Path | None = None, arm: str | None = None,
                             quality_bank: Mapping[str, Any] | None = None
                             ) -> dict[str, Any]:
    """The REAL per-candidate quality metric computation.

    Mirrors the historical, production C6 measurement path EXACTLY --
    `pipeline.adapters.c6._evaluate_generated_candidates` ->
    `synthesis.c6_scientific.evaluate_pool`, which reconstructs each finalized
    candidate's bytes (`c6_scientific.reconstruct_discrete`, never re-renders),
    rebuilds the requested support mask the same deterministic way the render
    routes did (`c6_scientific.requested_support_for`), and measures it once
    with the canonical, frozen evaluator (`synthetic_bank.CandidateEvaluator`,
    itself `synthesis.quality_models`' three pinned backends wrapped by
    `quality_calibration.QualityBackends`). `c6_scientific.raw_metrics_of`
    unwraps exactly the raw, THRESHOLD-INDEPENDENT metric fields
    `quality_gate.evaluate` requires -- never the embedded gate decision the
    evaluator's own call also computes (discarded here, same as historical C6:
    `default_quality_matcher` gates with its OWN resolved profile thresholds).

    `candidates_root`/`arm` are OPTIONAL, explicit, caller-supplied overrides
    of WHERE the candidate's persisted bytes live -- pure storage-location
    metadata, never a scientific input. Omitting them (the historical E6
    call path always omits them) preserves the EXACT historical default
    (`CANDIDATES_ROOT`/`E6_ARM_NAME`) byte-for-byte; a caller that supplies
    them (E6-v2) is never silently redirected back to that historical
    default -- this was ATTEMPT-1's proven second bug (a v2 quality lookup
    resolving the historical E6 root regardless of which v2 arm was being
    measured).

    `quality_bank` is likewise an OPTIONAL, explicit, caller-supplied
    override of WHICH recipe bank `requested_support_for`/
    `reconstruct_discrete` reconstruct this candidate's requested support
    mask/strength against. Omitting it preserves the exact historical
    default (`_resolve_quality_bank`, the LLM-SHUFFLE-A bank) byte-for-byte;
    a caller that supplies it (E6-v2, one bank per arm, built from that
    arm's OWN recipes) is never silently measured against a mismatched
    bank's recipe CONTENT -- this was ATTEMPT-2's proven third bug.

    Every test injects a fake `metrics_provider` OR monkeypatches
    `_resolve_quality_runtime`; this function is never invoked against real
    model weights on this laptop.
    """
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c6_scientific

    if not record.get("reusable", True):
        raise E6RenderError(
            f"{row.get('candidate_id')}: cannot measure quality for a candidate the renderer did not "
            f"produce (record={record!r})")

    runtime = _resolve_quality_runtime(repo, quality_bank=quality_bank)
    resolved_candidates_root = candidates_root if candidates_root is not None else repo / CANDIDATES_ROOT
    resolved_arm = arm if arm is not None else E6_ARM_NAME
    directory = raw.candidate_dir(resolved_candidates_root, resolved_arm, row["candidate_id"])
    stored = raw.read_record(directory / raw.RECORD_NAME)
    if stored is None or stored.get("status") != raw.GENERATED:
        raise E6RenderError(
            f"{row['candidate_id']}: no GENERATED candidate record at {directory.as_posix()}; "
            "quality metrics require the actual rendered payload bytes")

    store, bank, evaluator = runtime["store"], runtime["bank"], runtime["evaluator"]
    original, _ = store.load(row["live_target_sample_id"])
    support, graph = c6_scientific.requested_support_for(store, bank, row)
    discrete = c6_scientific.reconstruct_discrete(directory, original)
    result = evaluator.evaluate(discrete, live_target_sample_id=row["live_target_sample_id"],
                                requested_strength=float(graph.nodes[0].strength),
                                requested_support=support)
    return c6_scientific.raw_metrics_of(result, row["candidate_id"])


def resolve_e6_route_quota(repo: Path) -> dict[str, dict[str, int]]:
    """The per-route, per-domain quota LLM-SHUFFLE-A must match.

    `synthesis.c6_matched_bank.common_capacity`/`route_quotas` MATHEMATICALLY
    require candidates from all three frozen arms (`set(by_arm) == set(ARMS)`,
    i.e. RND, DET, LLM) to compute a shared quota vector -- see
    `E6_C6_MATCHING_TRACE.json`. Feeding LLM-SHUFFLE-A into that computation
    (as a 4th arm, or in place of LLM) would either raise or silently
    recompute a quota that could retroactively change how the FROZEN
    historical RND/DET/LLM banks are described. Neither is acceptable.

    The scientifically correct adaptation: LLM-SHUFFLE-A reuses the quota
    ORIGINAL_LLM's OWN historical bank already achieved (read directly from
    the frozen `C6_BANK_LOCK_LLM.json`), never recomputed jointly. If
    LLM-SHUFFLE-A's own eligible candidates cannot fill that SAME quota,
    `select_route_bank` raises -- a genuine finding, never routed around.
    """
    bank_lock = cc.read_json(repo / C6_BANK_LOCK_LLM_PATH)
    quota: dict[str, dict[str, int]] = {}
    for route, exposure in bank_lock["exposure"].items():
        quota[route] = dict(exposure["by_source_domain"])
    return quota


def default_quality_matcher(*, repo: Path, plan: dict[str, Any], staged: dict[str, Any], arm: str,
                            metrics_provider: Callable[..., dict[str, Any]] | None = None,
                            candidates_root: Path | None = None,
                            quality_bank: Mapping[str, Any] | None = None
                            ) -> dict[str, Any]:
    """The REAL quality-score + gate + C6-matching pass, reusing
    `synthesis.quality_gate.Thresholds`/`evaluate` and
    `synthesis.c6_matched_bank.SelectableCandidate`/`select_route_bank`/
    `selected_set_digest` VERBATIM -- never a reimplementation. The quota is
    resolved from ORIGINAL_LLM's own frozen bank (`resolve_e6_route_quota`),
    never recomputed jointly across arms (see that function's docstring).
    Per-candidate metric computation is the one remaining GPU-only seam
    (`metrics_provider`); every test injects a fake one.

    `arm` is a REQUIRED, explicit caller-supplied label -- it is BINDING/
    METADATA only: it is written verbatim into every `SelectableCandidate`
    this builds and therefore into `selected`'s serialized rows, but it never
    enters the quality-gate decision, the route quota, the selection
    objective, the ordering/tie-breaking inside `select_route_bank`, or the
    bank size -- and `selected_set_digest` deliberately hashes only
    `selection_step:candidate_id` (never `arm`, per that function's own
    docstring: "identity over WHICH candidates were selected"), so
    `selected_set_sha256` itself is IDENTICAL across two calls that differ
    only in `arm`. Two calls with identical `repo`/`plan`/`staged`/
    `metrics_provider` and different `arm` values therefore select the exact
    same candidates, in the exact same order, by the exact same rule,
    differing ONLY in how each selected row's own `arm` field reads. There is
    deliberately no default -- every caller must say, explicitly, which arm
    it is matching for (historical E6 passes `E6_ARM_NAME`; E6-v2 passes one
    of its own two new arm labels).

    `candidates_root`/`quality_bank` are likewise OPTIONAL, explicit,
    storage-location and reconstruction-bank overrides forwarded verbatim to
    `default_metrics_provider` (never to a custom injected
    `metrics_provider`, whose signature this must not assume). Omitting
    them preserves the historical E6 default exactly (the historical
    candidate root, and the LLM-SHUFFLE-A bank); a caller that supplies them
    (E6-v2, one bank/root per arm) is never silently redirected back to
    that historical default -- `quality_bank` closes ATTEMPT-2's proven
    third bug (every arm's candidates being reconstructed against
    LLM-SHUFFLE-A's recipe CONTENT regardless of which arm actually
    rendered them).
    """
    from prism_fas.synthesis.c6_matched_bank import SelectableCandidate, select_route_bank, selected_set_digest
    from prism_fas.synthesis.quality_gate import Thresholds

    provider = metrics_provider or default_metrics_provider
    provider_location_kwargs = ({"candidates_root": candidates_root, "arm": arm, "quality_bank": quality_bank}
                                if provider is default_metrics_provider else {})
    gate_profiles = cc.read_json(repo / C6_GATE_PROFILES_PATH)
    thresholds = Thresholds.from_dict(gate_profiles["profiles"][EXPECTED_QUALITY_PROFILE]["thresholds"])
    quota = resolve_e6_route_quota(repo)

    eligible_by_route: dict[str, list[SelectableCandidate]] = {"physics": [], "gpat": []}
    q_by_candidate: dict[str, float] = {}
    for row, result in zip(staged["rows"], staged["results"]):
        metrics = provider(repo=repo, row=row, record=result, **provider_location_kwargs)
        from prism_fas.synthesis.quality_gate import evaluate

        decision = evaluate(metrics, thresholds)
        if not decision["accepted"]:
            continue
        q_by_candidate[row["candidate_id"]] = decision["q"]
        eligible_by_route[row["route"]].append(SelectableCandidate(
            candidate_id=row["candidate_id"], arm=arm, route=row["route"],
            source_domain=row.get("live_dataset", ""), recipe_id=row["recipe_id"],
            recipe_ordinal=row["recipe_ordinal"], live_target_sample_id=row["live_target_sample_id"],
            base_position=row["position"], q=decision["q"]))

    selected: list[dict[str, Any]] = []
    for route in ("physics", "gpat"):
        rows = select_route_bank(eligible_by_route[route], route=route, quota=quota[route])
        selected.extend(rows)

    return {"selected": selected, "selected_set_sha256": selected_set_digest(selected),
           "original_llm_q": _read_original_llm_q_reference(repo)}


def _read_original_llm_q_reference(repo: Path) -> dict[str, Any]:
    """TASK H: the EXACT historical ORIGINAL_LLM q statistics, read from the
    frozen reconstructed-q artifact -- never the approximate prose values."""
    path = repo / "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv"
    if not path.is_file():
        raise E6RenderError(f"missing frozen ORIGINAL_LLM q reference artifact at {path.as_posix()}")
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "LLM":
                return {"n": int(row["n"]), "mean": float(row["q_mean"]), "sample_sd": float(row["q_sd_ddof1"]),
                       "median": float(row["q_median"]), "q1": float(row["q_q1"]), "q3": float(row["q_q3"]),
                       "source_artifact": "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv",
                       "source_artifact_sha256": cc.sha256_file(path)}
    raise E6RenderError("FINAL_Q_SUMMARY.csv has no LLM row")


# --------------------------------------------------------------------------- #
# G. Historical q reproduction validator -- never fabricates PASS
# --------------------------------------------------------------------------- #

def _resolve_historical_llm_bank(repo: Path) -> dict[str, Any]:
    """The `bank` shape needed to recompute a HISTORICAL ORIGINAL_LLM
    candidate: built from the real, frozen, UNSHUFFLED LLM recipe bank
    (`assets/recipe_banks/c3/llm/recipes.jsonl`) -- deliberately NOT the
    LLM-SHUFFLE-A bank `_resolve_quality_bank` builds. A historical candidate's
    `recipe_id` is, by construction, IDENTICAL under the shuffle
    (`verify_source_pair_recipe_alignment`), but its recipe field CONTENT is
    not -- looking a historical candidate's recipe_id up in the shuffled bank
    would silently compile the WRONG graph (wrong region mask, wrong requested
    strength) for it.
    """
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    original_recipes = cc.read_jsonl(repo / "assets/recipe_banks/c3/llm/recipes.jsonl")
    ontology = load_ontology(repo / ONTOLOGY_CONFIG_PATH)
    if ontology.sha256 != EXPECTED_ONTOLOGY_IDENTITY:
        raise E6RenderError(
            f"resolved ontology identity {ontology.sha256!r} != pinned {EXPECTED_ONTOLOGY_IDENTITY!r}")
    bank_lock_path = repo / "assets/recipe_banks/c3/llm/C3_BANK.json"
    bank_identity = None
    if bank_lock_path.is_file():
        bank_identity = str(json.loads(bank_lock_path.read_text(encoding="utf-8")).get("bank_identity") or "") or None
    parsed = [parse_recipe(recipe) for recipe in original_recipes]
    return {"recipes": parsed, "bank_id": "c3_llm", "ontology": ontology, "bank_identity": bank_identity,
           "ontology_identity": ontology.sha256}


def run_historical_q_reproduction(repo: Path, *, sample_size: int = 32,
                                  runtime: dict[str, Any] | None = None,
                                  historical_bank: dict[str, Any] | None = None) -> dict[str, Any]:
    """Real GPU-forward validation: for a deterministic sample of the frozen
    ORIGINAL_LLM SELECTED candidates (`c_ext_quality_reconstruct.extract_selected_q`'s
    Path-A rows -- the only ones with a persisted ground-truth q, verbatim from
    `C6_BANK_LOCK_LLM.json`, no reconstruction), recomputes each candidate's raw
    metrics with the SAME canonical chain `default_metrics_provider` uses
    (`c6_scientific.reconstruct_discrete` / `requested_support_for` +
    `synthetic_bank.CandidateEvaluator`) and compares the recomputed q against
    the persisted one. Read-only: opens `runs/full/c5/scientific/candidates`
    and `reports/full/c6/C6_BANK_LOCK_LLM.json`, writes nothing, never touches
    a historical artifact.
    """
    from prism_fas.evaluation.c_ext_quality_reconstruct import extract_selected_q
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c6_scientific

    rows, _ = extract_selected_q(repo)
    llm_rows = sorted((row for row in rows if row.arm == "LLM"), key=lambda row: row.candidate_id)
    llm_rows = llm_rows[:sample_size]
    if not llm_rows:
        raise E6RenderError("no frozen ORIGINAL_LLM selected rows available to validate against")

    runtime = runtime or _resolve_quality_runtime(repo)
    bank = historical_bank or _resolve_historical_llm_bank(repo)
    store, evaluator = runtime["store"], runtime["evaluator"]
    candidates_root = repo / HISTORICAL_LLM_CANDIDATE_ROOT

    checked: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for historical in llm_rows:
        directory = raw.candidate_dir(candidates_root, "LLM", historical.candidate_id)
        stored = raw.read_record(directory / raw.RECORD_NAME)
        if stored is None or stored.get("status") != raw.GENERATED:
            raise E6RenderError(f"{historical.candidate_id}: missing GENERATED historical candidate record")
        live_target_sample_id = stored["generation_identity"]["live_target_sample_id"]
        recipe_id = stored["generation_identity"]["recipe_id"]
        plan_row = {"candidate_id": historical.candidate_id, "recipe_id": recipe_id,
                   "live_target_sample_id": live_target_sample_id}
        original, _ = store.load(live_target_sample_id)
        support, graph = c6_scientific.requested_support_for(store, bank, plan_row)
        discrete = c6_scientific.reconstruct_discrete(directory, original)
        outcome = evaluator.evaluate(discrete, live_target_sample_id=live_target_sample_id,
                                     requested_strength=float(graph.nodes[0].strength),
                                     requested_support=support)
        recomputed_q = float(outcome["q"])
        entry = {"candidate_id": historical.candidate_id, "historical_q": historical.q,
                 "recomputed_q": recomputed_q, "abs_diff": abs(recomputed_q - historical.q)}
        checked.append(entry)
        if entry["abs_diff"] > 1e-6:
            mismatches.append(entry)

    return {"schema_version": "e6-historical-q-reproduction-v1",
           "historical_q_reproduction_executed": True,
           "historical_q_reproduction_status": "PASS" if not mismatches else "FAIL",
           "checked": len(checked), "mismatches": mismatches[:16],
           "mismatch_count": len(mismatches)}


def historical_q_reproduction_status(repo: Path, *, sample_size: int = 32) -> dict[str, Any]:
    """TASK G: whether historical metrics + `quality_gate.evaluate` can
    reproduce persisted historical q. Never fabricates PASS: only actually
    RUNS `run_historical_q_reproduction` when BOTH the real quality backend
    stack AND the real historical candidate payload bytes resolve on this
    host; reports `DEFERRED`, with the exact missing dependency named,
    otherwise (expected on this laptop -- both are GPU-host-only)."""
    assets = resolve_quality_backend_assets(repo)
    candidate_root = repo / HISTORICAL_LLM_CANDIDATE_ROOT / "LLM"
    payload_resolvable = candidate_root.is_dir() and any(candidate_root.iterdir())
    base = {
        "schema_version": "e6-historical-q-reproduction-status-v1",
        "historical_q_reproduction_implemented": True,
        "quality_backends_resolvable": assets["QUALITY_BACKENDS_RESOLVABLE"],
        "historical_candidate_payload_resolvable": payload_resolvable,
        "historical_candidate_root": HISTORICAL_LLM_CANDIDATE_ROOT,
    }
    if not (assets["QUALITY_BACKENDS_RESOLVABLE"] and payload_resolvable):
        return {**base, "historical_q_reproduction_executed": False,
               "historical_q_reproduction_status": "DEFERRED",
               "reason": "quality backend weights and/or the real historical rendered candidate "
                         "payload bytes are not present on this host"}
    return {**base, **run_historical_q_reproduction(repo, sample_size=sample_size)}


def gpu_historical_q_audit_command() -> str:
    """TASK G/H: prepared, never executed here. Read-only on the GPU host."""
    return "python -m prism_fas.evaluation.c_ext_e6_render --historical-q-audit"


# --------------------------------------------------------------------------- #
# Per-candidate quality diagnostic (TASK C/D/F) -- read-only, no render, no
# duplicate quality implementation. Reuses the SAME canonical chain
# default_metrics_provider / run_historical_q_reproduction already use.
# --------------------------------------------------------------------------- #

#: The raw metrics quality_gate.evaluate consults, mapped to their threshold
#: field, the hard gate name evaluate() records them under, and the
#: quality_components key evaluate() derives from them -- read from the
#: canonical quality_gate module's OWN documented shape, never invented.
_METRIC_GATE_MAP: dict[str, tuple[str | None, str | None, str | None]] = {
    "face_detection_score": ("tau_fd", "face_detection", "q_fd"),
    "identity_cosine": ("tau_id", "identity", "q_id"),
    "landmark_nme": ("tau_lm", "landmark", "q_lm"),
    "outside_mask_parsing_dice": ("tau_parse", "parsing_dice", "q_parse"),
    "outside_mask_max_error": ("tau_out", "outside_mask", None),
    "measured_artifact_strength": (None, "artifact_strength", "q_strength"),
    "requested_artifact_strength": (None, "artifact_strength", "q_strength"),
    "fingerprint_score": ("tau_fp", "fingerprint", "q_fp"),
    "support_overlap": (None, "support_overlap", "q_support"),
    "reference_detection_score": (None, None, None),
    "landmark_detected": (None, None, None),
}


def _sanitize_diagnostic_error(reason: str) -> str:
    import re

    text = re.sub(r"[A-Za-z]:[\\/][^\s'\"]*", "<path>", str(reason))
    text = re.sub(r"(/[^\s'\"/]+)+/", "<path>/", text)
    return text[:400]


def diagnose_historical_candidate(repo: Path, candidate_id: str, *, arm: str = "LLM",
                                  runtime: dict[str, Any] | None = None,
                                  historical_bank: dict[str, Any] | None = None
                                  ) -> dict[str, Any]:
    """TASK C: a READ-ONLY, per-candidate quality diagnostic for ONE frozen
    ORIGINAL_LLM (or, with `arm=`, RND/DET) selected candidate.

    Renders NOTHING: the candidate's bytes are RECONSTRUCTED from the frozen
    on-disk payload (`c6_scientific.reconstruct_discrete`), exactly the same
    real reconstruction `run_historical_q_reproduction` already performs for a
    SAMPLE -- this is the SAME chain for exactly ONE candidate, with every raw
    metric, gate and component surfaced instead of only `q`. Uses the SAME
    canonical `QualityBackends` / `CandidateEvaluator` / `quality_gate.evaluate`
    -- never a second quality implementation, never a duplicated formula.
    """
    from prism_fas.evaluation.c_ext_quality_reconstruct import extract_selected_q
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c6_scientific

    rows, _ = extract_selected_q(repo)
    historical = next((row for row in rows if row.candidate_id == candidate_id and row.arm == arm), None)
    if historical is None:
        raise E6RenderError(
            f"{candidate_id}: not a frozen ORIGINAL_LLM selected candidate (arm={arm}); "
            "the historical-q oracle (extract_selected_q) has no record of it")

    candidates_root = repo / HISTORICAL_LLM_CANDIDATE_ROOT
    directory = raw.candidate_dir(candidates_root, arm, candidate_id)
    stored = raw.read_record(directory / raw.RECORD_NAME)
    if stored is None or stored.get("status") != raw.GENERATED:
        raise E6RenderError(
            f"{candidate_id}: no GENERATED historical candidate record at {directory.as_posix()}")
    identity = stored["generation_identity"]
    route = identity["route"]
    live_id = identity["live_target_sample_id"]
    spoof_id = identity.get("spoof_source_sample_id")
    payload_sha256 = dict(stored.get("payload_sha256") or {})

    runtime = runtime or _resolve_quality_runtime(repo)
    bank = historical_bank or _resolve_historical_llm_bank(repo)
    store, evaluator = runtime["store"], runtime["evaluator"]

    # TASK D: reference sample resolution -- read-only file hashes, source-pair
    # mapping never touched. CandidateEvaluator.evaluate's own signature
    # (`live_target_sample_id`, no `spoof_source_sample_id` parameter) proves
    # only the LIVE reference is read at MEASUREMENT time; the spoof source
    # was consumed only at GENERATION time (GPAT texture mixing), already
    # baked into the rendered pixels this reconstructs from bytes.
    live_row = store.row(live_id)
    live_path = store.package_root / live_row["image_relative_path"]
    live_sha256 = cc.sha256_file(live_path) if live_path.is_file() else None
    spoof_resolution: dict[str, Any] | None = None
    if spoof_id:
        try:
            spoof_row = store.row(spoof_id)
            spoof_path = store.package_root / spoof_row["image_relative_path"]
            spoof_resolution = {
                "spoof_source_sample_id": spoof_id,
                "resolved_spoof_path": str(spoof_path),
                "resolved_spoof_file_sha256": cc.sha256_file(spoof_path) if spoof_path.is_file() else None,
                "used_by_candidate_evaluator_at_measurement_time": False,
            }
        except Exception as error:  # noqa: BLE001 - diagnostic must never crash on a resolution failure
            spoof_resolution = {"spoof_source_sample_id": spoof_id,
                                "error": _sanitize_diagnostic_error(str(error))}

    original, _ = store.load(live_id)
    plan_row = {"candidate_id": candidate_id, "recipe_id": identity["recipe_id"],
               "live_target_sample_id": live_id}
    support, graph = c6_scientific.requested_support_for(store, bank, plan_row)
    discrete = c6_scientific.reconstruct_discrete(directory, original)
    outcome = evaluator.evaluate(discrete, live_target_sample_id=live_id,
                                 requested_strength=float(graph.nodes[0].strength), requested_support=support)
    raw_metrics = c6_scientific.raw_metrics_of(outcome, candidate_id)
    recomputed_q = float(outcome["q"])
    historical_q = float(historical.q)

    # TASK C/D forensic: compares the FULL region-mask result recomputed NOW
    # (synthetic_bank._support_masks, the SAME function requested_support_for
    # calls internally, but its full RegionMaskResult -- region_sources, per-
    # region pixel counts, achieved coverage, mask hash -- discarded by
    # requested_support_for's stripped boolean-array return) against the
    # SCALAR geometry summary render_one ALREADY PERSISTED into this
    # candidate's own CANDIDATE.json trace at GENERATION time
    # (requested_support_pixels / requested_coverage / achieved_coverage --
    # see c5_render.render_one). A mismatch here is DIRECT, CONCRETE evidence
    # that the region/coverage recompute diverges from what was actually used
    # to build exact_mask.png, narrowing the root cause to the mask-building
    # step specifically -- never inferred, always read from real persisted
    # values and a real recompute.
    import numpy as np

    from prism_fas.synthesis import synthetic_bank

    # Hoisted out of the try block below: depends only on `discrete`, which is
    # already real by this point, so it stays available even if the mask
    # recompute itself fails -- both the mask_forensics fallback and the
    # alternative-input reconstruction below need it. Guarded: a test double
    # (or a genuinely malformed reconstruction) may not carry the real shape.
    try:
        exact_pixels_now = int(np.asarray(discrete.exact_edit_mask).astype(bool).sum())
    except Exception:  # noqa: BLE001 - forensic must never crash the diagnostic
        exact_pixels_now = None

    mask_forensics: dict[str, Any] = {}
    try:
        full_mask_result = synthetic_bank._support_masks(store, live_id, graph)
        generation_trace = dict(stored.get("trace") or {})
        recomputed_support_pixels = int(np.asarray(full_mask_result.operator_support_mask).astype(bool).sum())
        generation_support_pixels = generation_trace.get("requested_support_pixels")
        mask_forensics = {
            "recomputed_now": {
                "requested_region_pixels": int(np.asarray(full_mask_result.requested_region_mask)
                                              .astype(bool).sum()),
                "support_pixels": recomputed_support_pixels,
                "requested_coverage": float(full_mask_result.requested_coverage),
                "achieved_coverage": float(full_mask_result.achieved_coverage),
                "region_sources": dict(full_mask_result.region_sources),
                "per_region_pixels": dict((full_mask_result.metadata or {}).get("per_region_pixels", {})),
                "coverage_within_tolerance": (full_mask_result.metadata or {}).get("coverage_within_tolerance"),
                "mask_hash": full_mask_result.mask_hash,
                "parsing_available": (full_mask_result.metadata or {}).get("parsing_available"),
            },
            "persisted_at_generation": {
                "requested_region_pixels": generation_trace.get("requested_region_pixels"),
                "requested_support_pixels": generation_support_pixels,
                "requested_coverage": generation_trace.get("requested_coverage"),
                "achieved_coverage": generation_trace.get("achieved_coverage"),
            },
            "support_pixel_count_matches_generation": (
                generation_support_pixels is not None
                and int(generation_support_pixels) == recomputed_support_pixels),
            "exact_mask_pixels_within_recomputed_support": None,  # filled below
        }
        overlap_pixels = int((np.asarray(discrete.exact_edit_mask).astype(bool)
                              & np.asarray(full_mask_result.operator_support_mask)[0].astype(bool)).sum())
        mask_forensics["exact_mask_pixels_now"] = exact_pixels_now
        mask_forensics["exact_within_recomputed_support_pixels"] = overlap_pixels
        mask_forensics["exact_mask_pixels_within_recomputed_support"] = (
            overlap_pixels == exact_pixels_now if exact_pixels_now else None)
    except Exception as error:  # noqa: BLE001 - forensic must never crash the diagnostic
        mask_forensics = {"error": _sanitize_diagnostic_error(str(error))}

    # TASK D forensic: recipe/bank IDENTITY comparison -- persisted at
    # GENERATION time (identity['recipe_bank_identity'], route_trace's
    # recipe_hash/graph_hash from GPATRoute.generate/PhysicsRoute.generate's
    # own trace dict) vs a FRESH compile_recipe of the SAME recipe_id read
    # from the CURRENTLY frozen recipes.jsonl. compile_recipe is a PURE
    # function of (recipe content, ontology, bank_id) -- if these hashes
    # disagree, the recipe CONTENT used at generation genuinely differed from
    # what is frozen in the repository today, despite sharing the same
    # recipe_id and despite git history showing no committed change (which
    # only proves no COMMITTED change, not that the working content at
    # generation time equalled today's committed content).
    try:
        route_trace = dict((stored.get("trace") or {}).get("route_trace") or {})
        persisted_bank_identity = identity.get("recipe_bank_identity")
        current_bank_identity = bank.get("bank_identity") if isinstance(bank, dict) else None
        persisted_recipe_hash = route_trace.get("recipe_hash")
        current_recipe_hash = getattr(graph, "recipe_hash", None)
        persisted_graph_hash = route_trace.get("graph_hash")
        current_graph_hash = getattr(graph, "graph_hash", None)
        identity_comparison = {
            "persisted_recipe_bank_identity": persisted_bank_identity,
            "current_bank_identity": current_bank_identity,
            "recipe_bank_identity_matches": (persisted_bank_identity is not None
                                             and current_bank_identity is not None
                                             and persisted_bank_identity == current_bank_identity),
            "persisted_recipe_hash": persisted_recipe_hash, "current_recipe_hash": current_recipe_hash,
            "recipe_hash_matches": (persisted_recipe_hash is not None
                                    and persisted_recipe_hash == current_recipe_hash),
            "persisted_graph_hash": persisted_graph_hash, "current_graph_hash": current_graph_hash,
            "graph_hash_matches": (persisted_graph_hash is not None
                                   and persisted_graph_hash == current_graph_hash),
            "recipe_geometry_coverage_now": round(float(getattr(graph, "region_mask_policy", {})
                                                        .get("requested_coverage")), 6),
        }
    except Exception as error:  # noqa: BLE001 - forensic must never crash the diagnostic
        identity_comparison = {"error": _sanitize_diagnostic_error(str(error))}

    # TASK C: alternative-input support reconstruction. Task A's static trace
    # proved ONLY live_target_sample_id is ever passed into _support_masks by
    # either route -- spoof_source_sample_id is never a `_support_masks`
    # input, only a GPATRoute model-conditioning input (`style` via
    # store.cached_mask). This EMPIRICALLY tests that finding rather than just
    # asserting it: recomputes the full RegionMaskResult using the SPOOF
    # sample's own priors in place of the live sample's, so a match against
    # the persisted generation trace would falsify the static conclusion.
    alternative_input_reconstructions: list[dict[str, Any]] = []
    for source_name, sample_id in (("live_target", live_id), ("spoof_source", spoof_id)):
        if sample_id is None:
            continue
        try:
            alt_row = store.row(sample_id)
            alt_path = store.package_root / alt_row["image_relative_path"]
            alt_sha256 = cc.sha256_file(alt_path) if alt_path.is_file() else None
            alt_result = synthetic_bank._support_masks(store, sample_id, graph)
            alt_support_pixels = int(np.asarray(alt_result.operator_support_mask).astype(bool).sum())
            alt_region_pixels = int(np.asarray(alt_result.requested_region_mask).astype(bool).sum())
            alt_overlap = int((np.asarray(discrete.exact_edit_mask).astype(bool)
                              & np.asarray(alt_result.operator_support_mask)[0].astype(bool)).sum())
            gen_region = (stored.get("trace") or {}).get("requested_region_pixels")
            gen_support = (stored.get("trace") or {}).get("requested_support_pixels")
            gen_coverage = (stored.get("trace") or {}).get("requested_coverage")
            alternative_input_reconstructions.append({
                "input_source": source_name, "input_sample_id": sample_id,
                "input_path": str(alt_path), "input_sha256": alt_sha256,
                "requested_region_pixels": alt_region_pixels, "support_pixels": alt_support_pixels,
                "requested_coverage": float(alt_result.requested_coverage),
                "achieved_coverage": float(alt_result.achieved_coverage),
                "mask_hash": alt_result.mask_hash,
                "MATCHES_GENERATION_REGION_PIXELS": (gen_region is not None
                                                     and int(gen_region) == alt_region_pixels),
                "MATCHES_GENERATION_SUPPORT_PIXELS": (gen_support is not None
                                                      and int(gen_support) == alt_support_pixels),
                "MATCHES_GENERATION_COVERAGE": (gen_coverage is not None
                                                and abs(float(gen_coverage)
                                                       - float(alt_result.requested_coverage)) < 1e-6),
                "EXACT_MASK_WITHIN_SUPPORT": alt_overlap == exact_pixels_now if exact_pixels_now else None,
                "SUPPORT_OVERLAP": (float(alt_overlap) / exact_pixels_now) if exact_pixels_now else 0.0,
            })
        except Exception as error:  # noqa: BLE001 - forensic must never crash the diagnostic
            alternative_input_reconstructions.append({"input_source": source_name, "input_sample_id": sample_id,
                                                      "error": _sanitize_diagnostic_error(str(error))})

    # TASK A/C: the VERBATIM persisted trace dict, unfiltered -- eliminates any
    # possibility of a field-provenance transcription error across turns/chat
    # relay. `mask_forensics`/`identity_comparison` above already read specific
    # keys out of this SAME dict; this exposes the whole thing so the next run
    # can be checked byte-for-byte against what this function actually parsed.
    raw_generation_trace = dict(stored.get("trace") or {})

    # TASK G: the region-mask seed is a PURE function of (bank_id, recipe_id,
    # recipe.seed, sample_id, node_index) -- all four already PROVEN identical
    # between generation and now whenever identity_comparison reports
    # graph_hash_matches=True (graph_hash's own hash material embeds bank_id,
    # recipe_id, recipe_seed and the node's own seed field). When that holds,
    # this computed value is not merely "the current seed" but PROVABLY the
    # historical seed too -- no separate historical record is needed to know it.
    region_mask_seed = None
    try:
        region_mask_seed = int(graph.node_seed(graph.nodes[0], f"{live_id}|region_mask"))
    except Exception:  # noqa: BLE001 - forensic must never crash the diagnostic
        pass

    # TASK F: the ONE thing this repository's package-identity chain does NOT
    # verify -- `content_identity_sha256` covers the SHARD TAR files
    # (data/package/builder.py: each shard's own sha256 feeds the package
    # lock hash), but `SampleStore.load()` reads LOOSE, unpacked files
    # directly from `images/`/`priors/` and never re-hashes them against
    # anything. This records a hash of the ACTUALLY-LOADED prior arrays for
    # live_id as a NEW baseline for future comparison -- it cannot retroactively
    # prove or disprove what generation read, only establish what THIS run
    # read, byte-for-byte, going forward.
    sample_priors_fingerprint = None
    try:
        _, live_arrays = store.load(live_id)
        hasher = hashlib.sha256()
        for key in sorted(live_arrays):
            hasher.update(key.encode("utf-8"))
            hasher.update(np.ascontiguousarray(live_arrays[key]).tobytes())
        sample_priors_fingerprint = hasher.hexdigest()
    except Exception:  # noqa: BLE001 - forensic must never crash the diagnostic
        pass

    return {
        "schema_version": "e6-candidate-quality-diagnostic-v1",
        "candidate_id": candidate_id, "arm": arm, "route": route,
        "recipe_id": identity.get("recipe_id"), "recipe_ordinal": identity.get("recipe_ordinal"),
        "slot": identity.get("slot"), "position": identity.get("position"),
        "live_target_sample_id": live_id,
        "resolved_live_path": str(live_path), "resolved_live_file_sha256": live_sha256,
        "spoof_source_resolution": spoof_resolution,
        "payload_sha256": payload_sha256,
        "synthetic_image_sha256": payload_sha256.get(raw.IMAGE_NAME),
        "artifact_map_sha256": payload_sha256.get(raw.ARTIFACT_MAP_NAME),
        "exact_mask_sha256": payload_sha256.get(raw.MASK_NAME),
        "raw_metrics": raw_metrics,
        "mask_forensics": mask_forensics,
        "identity_comparison": identity_comparison,
        "alternative_input_reconstructions": alternative_input_reconstructions,
        "raw_generation_trace": raw_generation_trace,
        "region_mask_seed": region_mask_seed,
        "sample_priors_fingerprint": sample_priors_fingerprint,
        "sample_priors_fingerprint_note": "sha256 over the ACTUALLY-LOADED parsing/landmarks/bbox/crop_box "
                                          "arrays for live_target_sample_id (sorted keys, raw array bytes). "
                                          "No historical baseline exists to compare against -- this "
                                          "establishes one for future runs; see TASK F in "
                                          "E6_SUPPORT_OVERLAP_ROOT_CAUSE.json for why SampleStore.load() "
                                          "never independently verifies loose prior files against the "
                                          "package's own content_identity_sha256 (which covers shard TAR "
                                          "files, not the loose files SampleStore actually reads).",
        "quality_thresholds": runtime["calibration"].thresholds.as_dict(),
        "quality_components": outcome.get("quality_components"),
        "failed_gates": outcome.get("failed_gates"),
        "accepted": outcome.get("accepted"),
        "recomputed_q": recomputed_q, "historical_q": historical_q,
        "abs_diff": abs(recomputed_q - historical_q),
        "historical_raw_metrics_available": False,
        "historical_raw_metrics_note": "no persisted per-candidate raw metric baseline exists anywhere "
                                       "in this repository (TASK E); the historical_q above is the ONLY "
                                       "persisted historical value, extracted verbatim from "
                                       "C6_BANK_LOCK_LLM.json via extract_selected_q -- never a "
                                       "reconstruction of raw metrics from q.",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }


def diagnostic_metric_table(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    """TASK F: flattens one diagnostic into the requested table shape --
    candidate_id / metric / recomputed value / historical value / threshold /
    gate status / q relevance. `historical value` is N/A for every raw metric
    except q itself (TASK E: no historical raw metrics are persisted)."""
    thresholds = diagnostic["quality_thresholds"]
    components = diagnostic.get("quality_components") or {}
    failed = set(diagnostic.get("failed_gates") or [])
    rows = [{"candidate_id": diagnostic["candidate_id"], "metric": "q",
            "recomputed_value": diagnostic["recomputed_q"], "historical_value": diagnostic["historical_q"],
            "threshold": None, "gate_status": "ACCEPTED" if diagnostic["accepted"] else "REJECTED",
            "q_component": None}]
    for metric, value in diagnostic["raw_metrics"].items():
        tau_key, gate_name, component_key = _METRIC_GATE_MAP.get(metric, (None, None, None))
        rows.append({
            "candidate_id": diagnostic["candidate_id"], "metric": metric, "recomputed_value": value,
            "historical_value": "N/A (not persisted; see historical_raw_metrics_available=False)",
            "threshold": thresholds.get(tau_key) if tau_key else None,
            "gate_status": ("FAIL" if gate_name in failed else "PASS") if gate_name else "N/A",
            "q_component": components.get(component_key) if component_key else None,
        })
    return rows


def diagnose_historical_candidates(repo: Path, candidate_ids: list[str], *, arm: str = "LLM"
                                   ) -> dict[str, Any]:
    """TASK F: diagnoses exactly the given candidate ids -- never sweeps the
    full 32-candidate audit again. Each candidate's failure (if any) is
    captured per-candidate; one bad id never aborts the batch. Never
    fabricates a result: if the real quality backend runtime cannot be
    resolved on this host (no CUDA, missing weights/calibration), reports
    DEFERRED for the whole batch rather than crashing with a raw traceback."""
    assets = resolve_quality_backend_assets(repo)
    if not assets["QUALITY_BACKENDS_RESOLVABLE"]:
        return {"schema_version": "e6-candidate-quality-diagnostic-batch-v1", "arm": arm,
               "requested": list(candidate_ids), "results": [],
               "diagnostic_executed": False, "diagnostic_status": "DEFERRED",
               "reason": "the real quality backend stack is not resolvable on this host "
                         "(QUALITY_BACKENDS_RESOLVABLE=False); see resolve_quality_backend_assets() "
                         "for the exact missing dependency",
               "target_access": False, "llm_api_calls": 0, "rendering_performed": False}

    try:
        runtime = _resolve_quality_runtime(repo)
        bank = _resolve_historical_llm_bank(repo)
    except Exception as error:  # noqa: BLE001 - a diagnostic must report, never crash, on a runtime failure
        return {"schema_version": "e6-candidate-quality-diagnostic-batch-v1", "arm": arm,
               "requested": list(candidate_ids), "results": [],
               "diagnostic_executed": False, "diagnostic_status": "FAIL",
               "reason": f"{type(error).__name__}: {_sanitize_diagnostic_error(str(error))}",
               "target_access": False, "llm_api_calls": 0, "rendering_performed": False}

    results = []
    for candidate_id in candidate_ids:
        try:
            diagnostic = diagnose_historical_candidate(
                repo, candidate_id, arm=arm, runtime=runtime, historical_bank=bank)
            results.append({"candidate_id": candidate_id, "ok": True, "diagnostic": diagnostic,
                           "table": diagnostic_metric_table(diagnostic)})
        except E6RenderError as error:
            results.append({"candidate_id": candidate_id, "ok": False, "error": str(error)})
    return {"schema_version": "e6-candidate-quality-diagnostic-batch-v1", "arm": arm,
           "requested": list(candidate_ids), "results": results,
           "diagnostic_executed": True, "diagnostic_status": "COMPLETE",
           "target_access": False, "llm_api_calls": 0, "rendering_performed": False}


def gpu_candidate_diagnostic_command(candidate_ids: list[str]) -> str:
    """TASK F: prepared, never executed here. Read-only on the GPU host."""
    ids = " ".join(candidate_ids)
    return f"python -m prism_fas.evaluation.c_ext_e6_render --diagnose-historical-candidate {ids}"


# --------------------------------------------------------------------------- #
# TASK G: shard-vs-loose-file byte audit -- read-only, prepared, distinguishes
# R4 (loose source prior byte drift) from R1/R2/R3. Never extracts to disk:
# tarfile member bytes are read into memory only and immediately hashed.
# --------------------------------------------------------------------------- #

#: Matches synthesis.data.package.shards.write_shard's own member convention
#: (source_train shard tar members are always <sample_id>.jpg/.npz/.json) --
#: reused as a documented assumption, not re-derived, since SampleStore's own
#: `image_relative_path` may point to a differently-suffixed loose file
#: (e.g. .png) that this audit compares AGAINST, not assumes equal.
SHARD_IMAGE_SUFFIX = ".jpg"
SHARD_PRIOR_SUFFIX = ".npz"


def shard_member_sha256(shard_path: Path, sample_id: str, suffix: str) -> str | None:
    """READ-ONLY: sha256 of one shard TAR member's bytes, read into memory --
    never extracted to disk, never writes anything. Returns None if the
    member is absent from this shard (a sample can live in only one shard)."""
    import tarfile

    name = f"{sample_id}{suffix}"
    with tarfile.open(shard_path, "r") as archive:
        try:
            member = archive.getmember(name)
        except KeyError:
            return None
        extracted = archive.extractfile(member)
        if extracted is None:
            return None
        return hashlib.sha256(extracted.read()).hexdigest()


def shard_vs_loose_byte_audit(repo: Path, sample_id: str, *, split: str = "source_train",
                              store: Any = None) -> dict[str, Any]:
    """TASK G: for ONE sample_id, compares the LOOSE file `SampleStore.load()`
    actually reads against the frozen SHARD TAR member the package's own
    `content_identity_sha256` hash chain covers (data/package/builder.py:
    each shard's own sha256 feeds the package lock hash; shard entries embed
    the raw image/prior bytes -- see E6_SUPPORT_OVERLAP_ROOT_CAUSE.json's
    TASK F finding). Distinguishes R4 (loose source prior byte drift) from
    R1/R2/R3: if loose == shard, the on-disk loose files genuinely are the
    bytes the package was built from; if they differ, the verification gap
    this milestone found is not merely theoretical for this sample.

    READ-ONLY: no extraction to disk, no repair, no overwrite. Prepared for
    the GPU host; the real M3B package is not present on this laptop.
    """
    from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit

    package_root = repo / SOURCE_PACKAGE_ROOT
    lock_path = package_root / "PACKAGE_LOCK.json"
    if not lock_path.is_file():
        return {"available": False, "sample_id": sample_id, "reason": f"missing {lock_path.as_posix()}"}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    shard_rows = [row for row in (lock.get("shards") or []) if row.get("split") == split]
    if not shard_rows:
        return {"available": False, "sample_id": sample_id,
               "reason": f"no shards recorded in PACKAGE_LOCK.json for split={split!r}"}

    try:
        store = store or SampleStore.open(package_root, SourceOnlyAudit())
        row = store.row(sample_id)
    except Exception as error:  # noqa: BLE001 - a preflight-style audit must never crash on missing data
        return {"available": False, "sample_id": sample_id,
               "reason": f"source_train manifest/sample not resolvable on this host: "
                         f"{type(error).__name__}: {_sanitize_diagnostic_error(str(error))}"}
    live_path = package_root / row["image_relative_path"]
    prior_path = package_root / row["prior_relative_path"]
    loose_image_sha256 = cc.sha256_file(live_path) if live_path.is_file() else None
    loose_prior_sha256 = cc.sha256_file(prior_path) if prior_path.is_file() else None

    matched_shard = None
    shard_image_sha256 = shard_prior_sha256 = None
    shard_archive_matches_package_lock = None
    for shard_row in shard_rows:
        shard_path = package_root / "shards" / str(shard_row["shard_filename"])
        if not shard_path.is_file():
            continue
        image_hash = shard_member_sha256(shard_path, sample_id, SHARD_IMAGE_SUFFIX)
        if image_hash is None:
            continue
        matched_shard = str(shard_row["shard_filename"])
        shard_image_sha256 = image_hash
        shard_prior_sha256 = shard_member_sha256(shard_path, sample_id, SHARD_PRIOR_SUFFIX)
        shard_archive_matches_package_lock = cc.sha256_file(shard_path) == shard_row.get("sha256")
        break

    return {
        "available": matched_shard is not None,
        "sample_id": sample_id, "matched_shard": matched_shard,
        "shard_archive_matches_package_lock": shard_archive_matches_package_lock,
        "loose_image_path": str(live_path), "loose_image_sha256": loose_image_sha256,
        "shard_image_sha256": shard_image_sha256,
        "image_matches": (loose_image_sha256 is not None and shard_image_sha256 is not None
                          and loose_image_sha256 == shard_image_sha256),
        "loose_prior_path": str(prior_path), "loose_prior_sha256": loose_prior_sha256,
        "shard_prior_sha256": shard_prior_sha256,
        "prior_matches": (loose_prior_sha256 is not None and shard_prior_sha256 is not None
                          and loose_prior_sha256 == shard_prior_sha256),
        "note": "None for a *_matches field means 'could not be checked' (missing file or shard), "
               "not 'checked and equal' -- never conflate the two.",
    }


# --------------------------------------------------------------------------- #
# TASK A-D: population-wide historical trace audit -- STRICTLY READ-ONLY.
# Recomputes ONLY recipe/graph hashes (pure compile_recipe, no image/model
# access whatsoever). Never instantiates QualityBackends, never touches
# SCRFD/AdaFace/FaceXFormer/GPAT, never renders, never trains, never accesses
# target. Every comparison tolerance is a MODULE CONSTANT, declared once,
# before any candidate is inspected -- never chosen after seeing results.
# --------------------------------------------------------------------------- #

#: Frozen BEFORE inspecting any population result. Matches the same
#: full-float64-vs-round(...,6) precision gap already established and used
#: (1e-6) for the single-candidate historical-q mismatch check
#: (run_historical_q_reproduction) and identity_comparison elsewhere in this
#: module -- reused verbatim for consistency, not re-derived per task.
TRACE_COMPARISON_TOLERANCE = 1e-6

#: The historical CANDIDATE.json record schema this module already reads
#: elsewhere (diagnose_historical_candidate) -- reused, not re-derived.
_POPULATION_AUDIT_MODEL_BACKENDS_INSTANTIATED: tuple[str, ...] = ()


def _close_within_tolerance(a: Any, b: Any, *, tolerance: float = TRACE_COMPARISON_TOLERANCE) -> bool | None:
    """None means 'not comparable' (a missing value), never False."""
    if a is None or b is None:
        return None
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return None


def audit_historical_trace_population(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASK A: a population-wide, READ-ONLY forensic pass over every frozen
    historical GENERATED CANDIDATE.json for one arm.

    For each candidate: reads the persisted CANDIDATE.json verbatim (never
    mutated), recompiles that candidate's recipe via the canonical
    `compile_recipe` (a PURE function of recipe+ontology+bank_id -- no image,
    no prior, no model, no GPU; recipes are compiled ONCE per unique
    recipe_id and cached, not once per candidate, matching the 256-unique-
    recipes-behind-2048-candidates structure), and compares the persisted
    recipe_hash/graph_hash/recipe_bank_identity against the freshly-recompiled
    ones. Reports every raw trace field TASK A asked for, plus the requested
    ratio/flag comparisons, using ONLY the module-level frozen tolerance.

    Reuses the SAME `_resolve_historical_llm_bank` this module's other
    diagnostics already use for arm='LLM' (the frozen, unshuffled recipe
    bank) -- for `arm='RND'`/`'DET'` this would need an analogous historical
    bank resolver, which does not exist in this module; those arms are out of
    this milestone's scope and raise a clear error rather than silently
    falling back to the LLM bank.
    """
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.synthesis import c5_raw_generation as raw

    if arm != "LLM":
        raise E6RenderError(
            f"audit_historical_trace_population only resolves the frozen LLM recipe bank "
            f"(_resolve_historical_llm_bank); arm={arm!r} has no analogous historical bank "
            f"resolver in this module -- refusing to guess one")

    candidates_root = repo / HISTORICAL_LLM_CANDIDATE_ROOT
    arm_root = candidates_root / arm
    if not arm_root.is_dir():
        return {"schema_version": "e6-historical-trace-population-v1", "available": False, "arm": arm,
               "reason": f"missing {arm_root.as_posix()} on this host", "row_count": 0, "rows": [],
               "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
               "training_performed": False,
               "model_backends_instantiated": list(_POPULATION_AUDIT_MODEL_BACKENDS_INSTANTIATED)}

    bank = _resolve_historical_llm_bank(repo)
    recipe_by_id = {recipe.recipe_id: recipe for recipe in bank["recipes"]}
    current_bank_identity = bank.get("bank_identity")
    graph_cache: dict[str, Any] = {}

    rows: list[dict[str, Any]] = []
    for candidate_dir in sorted(p for p in arm_root.iterdir() if p.is_dir()):
        record_path = candidate_dir / raw.RECORD_NAME
        if not record_path.is_file():
            continue
        stored = raw.read_record(record_path)
        if stored is None or stored.get("status") != raw.GENERATED:
            continue  # semantic failures carry no trace to audit; retained provenance, not touched here
        identity = stored.get("generation_identity") or {}
        trace = dict(stored.get("trace") or {})
        route_trace = dict(trace.get("route_trace") or {})
        recipe_id = identity.get("recipe_id")
        candidate_id = identity.get("candidate_id")

        recipe = recipe_by_id.get(recipe_id)
        if recipe is None:
            rows.append({"candidate_id": candidate_id, "route": identity.get("route"), "recipe_id": recipe_id,
                        "error": "recipe_id not found in the current frozen LLM recipe bank"})
            continue
        if recipe_id not in graph_cache:
            graph_cache[recipe_id] = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        graph = graph_cache[recipe_id]

        persisted_recipe_hash = route_trace.get("recipe_hash")
        persisted_graph_hash = route_trace.get("graph_hash")
        persisted_bank_identity = identity.get("recipe_bank_identity")
        current_recipe_hash = graph.recipe_hash
        current_graph_hash = graph.graph_hash
        recipe_geometry_coverage = (graph.region_mask_policy or {}).get("requested_coverage")

        trace_requested_coverage = trace.get("requested_coverage")
        trace_achieved_coverage = trace.get("achieved_coverage")
        trace_requested_region_pixels = trace.get("requested_region_pixels")
        trace_requested_support_pixels = trace.get("requested_support_pixels")
        trace_exact_mask_pixels = trace.get("exact_mask_pixels")

        support_over_region = None
        if (trace_requested_region_pixels not in (None, 0)
                and trace_requested_support_pixels is not None):
            support_over_region = float(trace_requested_support_pixels) / float(trace_requested_region_pixels)

        rows.append({
            "candidate_id": candidate_id, "route": identity.get("route"), "recipe_id": recipe_id,
            "recipe_ordinal": identity.get("recipe_ordinal"), "slot": identity.get("slot"),
            "position": identity.get("position"), "live_target_sample_id": identity.get("live_target_sample_id"),
            "persisted_recipe_hash": persisted_recipe_hash, "current_recipe_hash": current_recipe_hash,
            "recipe_hash_matches": (persisted_recipe_hash == current_recipe_hash
                                    if persisted_recipe_hash is not None else None),
            "persisted_graph_hash": persisted_graph_hash, "current_graph_hash": current_graph_hash,
            "graph_hash_matches": (persisted_graph_hash == current_graph_hash
                                   if persisted_graph_hash is not None else None),
            "persisted_recipe_bank_identity": persisted_bank_identity,
            "current_recipe_bank_identity": current_bank_identity,
            "bank_identity_matches": (persisted_bank_identity == current_bank_identity
                                      if persisted_bank_identity is not None and current_bank_identity is not None
                                      else None),
            "recipe_geometry_coverage": recipe_geometry_coverage,
            "trace_requested_coverage": trace_requested_coverage,
            "trace_achieved_coverage": trace_achieved_coverage,
            "trace_requested_region_pixels": trace_requested_region_pixels,
            "trace_requested_support_pixels": trace_requested_support_pixels,
            "trace_exact_mask_pixels": trace_exact_mask_pixels,
            "support_over_region": support_over_region,
            "requested_equals_recipe_coverage": _close_within_tolerance(trace_requested_coverage,
                                                                        recipe_geometry_coverage),
            "requested_equals_achieved_coverage": _close_within_tolerance(trace_requested_coverage,
                                                                          trace_achieved_coverage),
            "requested_equals_support_ratio": _close_within_tolerance(trace_requested_coverage,
                                                                      support_over_region),
            "achieved_equals_support_ratio": _close_within_tolerance(trace_achieved_coverage,
                                                                     support_over_region),
            "route_binding": trace.get("binding"),
            "candidate_json_sha256": cc.sha256_file(record_path),
        })

    return {"schema_version": "e6-historical-trace-population-v1", "available": True, "arm": arm,
           "tolerance": TRACE_COMPARISON_TOLERANCE, "row_count": len(rows), "rows": rows,
           "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
           "training_performed": False,
           "model_backends_instantiated": list(_POPULATION_AUDIT_MODEL_BACKENDS_INSTANTIATED)}


#: The 4 route-level match-count fields TASK B asks for, paired with the row
#: key each is computed FROM. Declared once so the aggregator and its tests
#: cannot silently drift apart on field names.
_AGGREGATE_MATCH_FIELDS: tuple[str, ...] = (
    "recipe_hash_matches", "graph_hash_matches", "bank_identity_matches",
    "requested_equals_recipe_coverage", "requested_equals_achieved_coverage",
    "requested_equals_support_ratio", "achieved_equals_support_ratio",
)
#: The 4 |a - b| deviation pairs TASK B asks summarized as mean/max abs.
_AGGREGATE_DEVIATION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("trace_requested_coverage", "recipe_geometry_coverage", "requested_vs_recipe"),
    ("trace_requested_coverage", "trace_achieved_coverage", "requested_vs_achieved"),
    ("trace_requested_coverage", "support_over_region", "requested_vs_support_ratio"),
    ("trace_achieved_coverage", "support_over_region", "achieved_vs_support_ratio"),
)


def aggregate_historical_trace_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK B: route-level (GPAT vs Physics) aggregate counts and mean/max
    absolute deviations -- descriptive forensic characterization only, no
    inferential statistics, exactly as instructed."""
    by_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "error" in row:
            continue
        by_route.setdefault(str(row.get("route")), []).append(row)

    aggregates: dict[str, Any] = {}
    for route, route_rows in sorted(by_route.items()):
        n = len(route_rows)
        counts = {field: sum(1 for row in route_rows if row.get(field) is True) for field in _AGGREGATE_MATCH_FIELDS}
        deviations: dict[str, Any] = {}
        for field_a, field_b, name in _AGGREGATE_DEVIATION_PAIRS:
            diffs = [abs(float(row[field_a]) - float(row[field_b])) for row in route_rows
                    if row.get(field_a) is not None and row.get(field_b) is not None]
            deviations[name] = {"n_comparable": len(diffs),
                               "mean_abs": (sum(diffs) / len(diffs)) if diffs else None,
                               "max_abs": max(diffs) if diffs else None}
        aggregates[route] = {"N": n, **{f"{field}_count": counts[field] for field in _AGGREGATE_MATCH_FIELDS},
                            "deviations": deviations}
    return {"schema_version": "e6-historical-trace-population-aggregate-v1", "by_route": aggregates,
           "contingency": {
               f"route={route}_x_requested_matches_recipe": aggregates[route]["requested_equals_recipe_coverage_count"]
               for route in aggregates
           }}


def classify_route_semantics_pattern(aggregates: dict[str, Any]) -> str:
    """TASK C: STRONG / PARTIAL / ABSENT, from a FROZEN rule declared here,
    never chosen after seeing the actual numbers:

    STRONG  -- for EVERY route with N>0, either (>=95% of candidates satisfy
               requested_equals_recipe_coverage) or (<=5% do), AND at least
               two routes disagree with each other by this same >=95%-vs-<=5%
               split. This is deliberately a STRICT threshold: a route-level
               pattern this mechanical and reproducible is what "systematic
               route semantics" would look like if real.
    PARTIAL -- routes disagree on their requested_equals_recipe_coverage
               rate, but not to the STRONG threshold above (i.e. a real but
               non-uniform difference, or one route is itself mixed).
    ABSENT  -- every route with N>0 has essentially the SAME
               requested_equals_recipe_coverage rate (within the same 5%
               band), or there is only one route with data.
    """
    by_route = aggregates.get("by_route") or {}
    rates: dict[str, float] = {}
    for route, agg in by_route.items():
        n = agg.get("N") or 0
        if n <= 0:
            continue
        rates[route] = agg["requested_equals_recipe_coverage_count"] / n

    if len(rates) < 2:
        return "ABSENT"

    near_one = [route for route, rate in rates.items() if rate >= 0.95]
    near_zero = [route for route, rate in rates.items() if rate <= 0.05]
    if near_one and near_zero and (len(near_one) + len(near_zero) == len(rates)):
        return "STRONG"

    spread = max(rates.values()) - min(rates.values())
    if spread >= 0.05:
        return "PARTIAL"
    return "ABSENT"


def write_historical_trace_population_artifacts(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASK G: writes the population CSV + summary JSON, ONLY under this
    extension's own render/ namespace, ONLY when the population is actually
    available on this host (never fabricates a population from nothing).
    Returns the same status dict either way."""
    import csv

    audit = audit_historical_trace_population(repo, arm=arm)
    out_dir = repo / RENDER_DIR
    csv_path = out_dir / "E6_HISTORICAL_TRACE_POPULATION.csv"
    summary_path = out_dir / "E6_HISTORICAL_TRACE_POPULATION_SUMMARY.json"

    if not audit["available"]:
        summary = {"schema_version": "e6-historical-trace-population-summary-v1", "available": False,
                  "arm": arm, "reason": audit.get("reason"),
                  "note": "the frozen historical candidate tree is not present on this host; no CSV "
                          "was written, and none was fabricated."}
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {**audit, "csv_written": False, "summary_written": True, "csv_path": None,
               "summary_path": str(summary_path)}

    rows = audit["rows"]
    fieldnames = [key for key in rows[0].keys()] if rows else []
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    aggregates = aggregate_historical_trace_population(rows)
    pattern = classify_route_semantics_pattern(aggregates)
    summary = {"schema_version": "e6-historical-trace-population-summary-v1", "available": True, "arm": arm,
              "tolerance": TRACE_COMPARISON_TOLERANCE, "row_count": len(rows),
              "aggregates": aggregates, "population_route_semantics_pattern": pattern,
              "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
              "training_performed": False,
              "model_backends_instantiated": list(_POPULATION_AUDIT_MODEL_BACKENDS_INSTANTIATED)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**audit, "csv_written": True, "summary_written": True, "csv_path": str(csv_path),
           "summary_path": str(summary_path), "aggregates": aggregates,
           "population_route_semantics_pattern": pattern}


def gpu_population_audit_command(*, arm: str = "LLM") -> str:
    """TASK A: prepared, never executed here. Read-only on the GPU host."""
    return f"python -m prism_fas.evaluation.c_ext_e6_render --audit-historical-trace-population --audit-arm {arm}"


# --------------------------------------------------------------------------- #
# TASK A-K (continuation turn): anomaly extraction, grouping, and integrity
# auditing over the population audit's OWN rows. Every function here is a
# pure, read-only transform of `rows` (as produced by
# audit_historical_trace_population) -- none opens a candidate file a second
# time, none touches an image/model/GPU, none writes anywhere but this
# extension's own render/ namespace.
# --------------------------------------------------------------------------- #

#: PLANNED candidates per route, per arm (C5_SOURCE_PAIR_PLAN.json:
#: candidates_per_arm=2048, 4 physics + 4 gpat per recipe x 256 recipes).
EXPECTED_PLANNED_PER_ROUTE = EXPECTED_CANDIDATES_PER_ARM // 2

#: The REAL, frozen, dated LLM-arm semantic-failure count
#: (reports/evidence/NEGATIVE_EVIDENCE_INDEX.json, entry
#: C5-SEMANTIC-GENERATION-FAILURES-2026-08: LLM generated=2034,
#: physics_failures=14, and "All 62 are Physics route; GPAT rendered 3072/3072"
#: across all three arms -- GPAT never semantically fails). Pinned from that
#: real artifact, not re-derived or guessed.
EXPECTED_LLM_PHYSICS_SEMANTIC_FAILURES = 14
EXPECTED_LLM_GPAT_GENERATED = EXPECTED_PLANNED_PER_ROUTE
EXPECTED_LLM_PHYSICS_GENERATED = EXPECTED_PLANNED_PER_ROUTE - EXPECTED_LLM_PHYSICS_SEMANTIC_FAILURES
EXPECTED_LLM_TOTAL_GENERATED = EXPECTED_LLM_GPAT_GENERATED + EXPECTED_LLM_PHYSICS_GENERATED

#: The 3 candidates the single-candidate historical-q audit (an earlier
#: continuation of this same investigation) already found and diagnosed in
#: depth. Flagged, never re-derived, in every anomaly row.
KNOWN_Q_AUDIT_MISMATCH_CANDIDATES: tuple[str, ...] = (
    "c5syn_0390812685e6403952baeb67", "c5syn_057aab8fef90ada42997a1a4", "c5syn_0588f57b2499484387d2b7af")


def population_integrity_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK H: route-count and duplicate-candidate_id reconciliation against
    the FROZEN, real expected counts (never assumed) -- catches a
    double-counted or otherwise inflated population BEFORE any anomaly
    analysis is trusted."""
    from collections import Counter

    usable = [row for row in rows if "error" not in row]
    route_counts = Counter(str(row.get("route")) for row in usable)
    id_counts = Counter(row.get("candidate_id") for row in usable)
    duplicates = {cid: n for cid, n in id_counts.items() if n > 1}
    expected = {"gpat": EXPECTED_LLM_GPAT_GENERATED, "physics": EXPECTED_LLM_PHYSICS_GENERATED}
    excess = {route: route_counts.get(route, 0) - expected.get(route, 0)
             for route in set(route_counts) | set(expected)}
    return {
        "schema_version": "e6-population-integrity-check-v1",
        "total_rows": len(usable), "rows_with_errors": len(rows) - len(usable),
        "observed_route_counts": dict(route_counts),
        "expected_route_counts": expected,
        "expected_total": EXPECTED_LLM_TOTAL_GENERATED,
        "excess_by_route": excess,
        "unique_candidate_id_count": len(id_counts),
        "duplicate_candidate_id_count": len(duplicates),
        "duplicate_candidate_ids": sorted(duplicates)[:50],
        "population_matches_expected": (not duplicates
                                        and all(value == 0 for value in excess.values())),
        "note": "EXPECTED_* is pinned from real, frozen artifacts (C5_SOURCE_PAIR_PLAN.json's "
               "candidates_per_arm=2048 split 4-physics+4-gpat per recipe, and "
               "NEGATIVE_EVIDENCE_INDEX.json's dated LLM physics_failures=14 with zero GPAT "
               "failures) -- never assumed. A nonzero excess or any duplicate candidate_id means "
               "the population audit result must be reconciled BEFORE its anomaly counts are used "
               "for further analysis.",
    }


def filter_anomalies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK A: exactly the rows where requested_equals_recipe_coverage is
    False (never None -- a missing value is 'not comparable', not an
    anomaly), enriched with the delta/ratio fields and the known-mismatch
    flag TASK A asked for."""
    anomalies = []
    for row in rows:
        if "error" in row or row.get("requested_equals_recipe_coverage") is not False:
            continue
        requested = row.get("trace_requested_coverage")
        achieved = row.get("trace_achieved_coverage")
        recipe = row.get("recipe_geometry_coverage")
        delta_requested_recipe = (requested - recipe) if requested is not None and recipe is not None else None
        delta_achieved_recipe = (achieved - recipe) if achieved is not None and recipe is not None else None
        delta_requested_achieved = (requested - achieved) if requested is not None and achieved is not None else None
        anomalies.append({
            **row,
            "delta_requested_recipe": delta_requested_recipe,
            "delta_achieved_recipe": delta_achieved_recipe,
            "delta_requested_achieved": delta_requested_achieved,
            "support_ratio": row.get("support_over_region"),
            "abs_delta_requested_recipe": (abs(delta_requested_recipe)
                                          if delta_requested_recipe is not None else None),
            "is_known_q_mismatch": row.get("candidate_id") in KNOWN_Q_AUDIT_MISMATCH_CANDIDATES,
        })
    return anomalies


def summarize_anomalies_by_key(anomaly_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]],
                               key_name: str) -> list[dict[str, Any]]:
    """TASK B/C/D: for every distinct value of `row[key_name]` seen among the
    anomalies, reports how many candidates sharing that key are anomalous vs
    the TOTAL candidates sharing it (from the full population), and which
    routes appear in each -- directly answers "does the same recipe/live
    sample/position produce BOTH normal and anomalous candidates" without
    assuming an answer."""
    from collections import defaultdict

    anomaly_by_key: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in anomaly_rows:
        anomaly_by_key[row.get(key_name)].append(row)
    total_by_key: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if "error" in row:
            continue
        total_by_key[row.get(key_name)].append(row)

    summary = []
    for key in sorted(anomaly_by_key, key=lambda value: (value is None, str(value))):
        arows = anomaly_by_key[key]
        trows = total_by_key.get(key, [])
        summary.append({
            "key_name": key_name, "key": key,
            "anomaly_count": len(arows), "total_count": len(trows),
            "anomaly_fraction": (len(arows) / len(trows)) if trows else None,
            "routes_in_anomalies": sorted({row.get("route") for row in arows}),
            "routes_in_all": sorted({row.get("route") for row in trows}),
            "key_is_uniformly_anomalous": len(arows) == len(trows) and len(trows) > 0,
        })
    return summary


def position_sequence_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK E: the full population ordered by frozen schedule position, with
    an explicit anomaly flag per row -- the raw material for contiguous-block/
    periodicity inspection. No per-candidate timestamp exists anywhere in the
    persisted trace or identity (verified: c5_raw_generation.GenerationIdentity
    and c5_render.render_one's trace dict carry no time field), so this is
    POSITION-ordered only; a timestamp-based change-point is NOT computable
    from any currently-persisted artifact."""
    usable = [row for row in rows if "error" not in row and row.get("position") is not None]
    ordered = sorted(usable, key=lambda row: int(row["position"]))
    return [{"position": int(row["position"]), "route": row.get("route"), "recipe_id": row.get("recipe_id"),
            "candidate_id": row.get("candidate_id"),
            "anomaly": row.get("requested_equals_recipe_coverage") is False} for row in ordered]


def find_contiguous_anomaly_blocks(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK E: runs of anomalous rows at CONSECUTIVE schedule positions (not
    merely adjacent in a possibly-gappy list index)."""
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in sequence:
        if item["anomaly"]:
            if current and item["position"] == current[-1]["position"] + 1:
                current.append(item)
            else:
                if current:
                    blocks.append(current)
                current = [item]
        else:
            if current:
                blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [{"start_position": block[0]["position"], "end_position": block[-1]["position"],
            "count": len(block), "candidate_ids": [item["candidate_id"] for item in block]} for block in blocks]


def find_max_deviation_candidates(rows: list[dict[str, Any]], *, field_a: str = "trace_requested_coverage",
                                  field_b: str = "recipe_geometry_coverage", top_n: int = 10
                                  ) -> list[dict[str, Any]]:
    """TASK F: the top-N candidates by |row[field_a] - row[field_b]|, to
    locate exactly which candidate(s) attain a reported maximum deviation."""
    scored = []
    for row in rows:
        if "error" in row:
            continue
        value_a, value_b = row.get(field_a), row.get(field_b)
        if value_a is None or value_b is None:
            continue
        scored.append((abs(float(value_a) - float(value_b)), row))
    scored.sort(key=lambda pair: -pair[0])
    return [{"abs_deviation": deviation, **row} for deviation, row in scored[:top_n]]


#: Frozen BEFORE inspecting any anomaly distribution -- TASK G's own
#: requested categories, in ascending order, last bucket catch-all.
_REQUESTED_VS_ACHIEVED_BUCKETS: tuple[tuple[float, str], ...] = (
    (1e-6, "<=1e-6"), (1e-4, "<=1e-4"), (1e-3, "<=1e-3"), (1e-2, "<=1e-2"), (float("inf"), ">1e-2"))


def categorize_requested_vs_achieved(rows: list[dict[str, Any]]) -> dict[str, int]:
    """TASK G: |requested - achieved| bucket counts, using the frozen
    boundaries above -- never chosen after seeing the distribution."""
    counts = {label: 0 for _, label in _REQUESTED_VS_ACHIEVED_BUCKETS}
    for row in rows:
        if "error" in row:
            continue
        requested, achieved = row.get("trace_requested_coverage"), row.get("trace_achieved_coverage")
        if requested is None or achieved is None:
            continue
        diff = abs(float(requested) - float(achieved))
        for bound, label in _REQUESTED_VS_ACHIEVED_BUCKETS:
            if diff <= bound:
                counts[label] += 1
                break
    return counts


def join_with_c6_bank_lock(rows: list[dict[str, Any]], bank_lock: dict[str, Any]) -> list[dict[str, Any]]:
    """TASK I: joins each row against the frozen C6_BANK_LOCK_LLM.json's own
    'selected' list (candidate_id -> q, route, ...) -- the ONLY per-candidate
    C6 outcome persisted anywhere for non-selected candidates. Never
    recomputes q. A candidate NOT in 'selected' has no persisted C6 outcome
    at all (accepted-but-not-selected and rejected are not distinguishable
    from any frozen artifact -- reported honestly as UNKNOWN, never guessed)."""
    selected_by_id = {str(row["candidate_id"]): row for row in (bank_lock.get("selected") or [])}
    joined = []
    for row in rows:
        selected_row = selected_by_id.get(row.get("candidate_id"))
        joined.append({
            **row,
            "c6_selected": selected_row is not None,
            "c6_selected_q": selected_row.get("q") if selected_row else None,
            "c6_accepted_or_rejected": ("SELECTED_IMPLIES_ACCEPTED" if selected_row is not None
                                        else "UNKNOWN (not persisted for non-selected candidates)"),
        })
    return joined


def write_anomaly_artifacts(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASK L: additive, read-only orchestration -- runs the population
    audit fresh (never trusts a stale CSV), computes the integrity check,
    filters anomalies, joins the frozen C6 bank lock where resolvable, and
    writes E6_HISTORICAL_TRACE_ANOMALIES.csv + _SUMMARY.json under this
    extension's own render/ namespace only. Never fabricates a result when
    the population itself is unavailable on this host."""
    audit = audit_historical_trace_population(repo, arm=arm)
    out_dir = repo / RENDER_DIR
    anomalies_csv_path = out_dir / "E6_HISTORICAL_TRACE_ANOMALIES.csv"
    summary_path = out_dir / "E6_HISTORICAL_TRACE_ANOMALY_SUMMARY.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audit["available"]:
        summary = {"schema_version": "e6-historical-trace-anomaly-summary-v1", "available": False,
                  "arm": arm, "reason": audit.get("reason"),
                  "note": "the frozen historical candidate tree is not present on this host; no "
                          "anomaly CSV was written, and none was fabricated."}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"available": False, "reason": audit.get("reason"), "csv_written": False,
               "summary_written": True, "csv_path": None, "summary_path": str(summary_path)}

    rows = audit["rows"]
    integrity = population_integrity_check(rows)
    anomalies = filter_anomalies(rows)

    bank_lock_path = repo / C6_BANK_LOCK_LLM_PATH
    if bank_lock_path.is_file() and arm == "LLM":
        bank_lock = json.loads(bank_lock_path.read_text(encoding="utf-8"))
        anomalies = join_with_c6_bank_lock(anomalies, bank_lock)
        c6_join_available = True
    else:
        c6_join_available = False

    import csv as csv_module

    fieldnames = list(anomalies[0].keys()) if anomalies else []
    with anomalies_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in anomalies:
            writer.writerow(row)

    by_recipe = summarize_anomalies_by_key(anomalies, rows, "recipe_id")
    by_live_sample = summarize_anomalies_by_key(anomalies, rows, "live_target_sample_id")
    by_position_mod_8 = summarize_anomalies_by_key(
        [{**row, "position_mod_8": (int(row["position"]) % EXPECTED_RENDERS_PER_RECIPE)
                                    if row.get("position") is not None else None} for row in anomalies],
        [{**row, "position_mod_8": (int(row["position"]) % EXPECTED_RENDERS_PER_RECIPE)
                                    if row.get("position") is not None else None} for row in rows],
        "position_mod_8")
    sequence = position_sequence_view(rows)
    blocks = find_contiguous_anomaly_blocks(sequence)
    max_deviation = find_max_deviation_candidates(rows, top_n=10)
    requested_vs_achieved_all = categorize_requested_vs_achieved(rows)
    requested_vs_achieved_anomalies = categorize_requested_vs_achieved(anomalies)
    known_mismatches_in_anomalies = [row for row in anomalies if row.get("is_known_q_mismatch")]

    summary = {
        "schema_version": "e6-historical-trace-anomaly-summary-v1", "available": True, "arm": arm,
        "integrity": integrity,
        "population_total": len([row for row in rows if "error" not in row]),
        "anomaly_count": len(anomalies),
        "normal_count": len([row for row in rows if "error" not in row]) - len(anomalies),
        "anomaly_route_counts": {route: sum(1 for row in anomalies if row.get("route") == route)
                                for route in sorted({row.get("route") for row in anomalies})},
        "unique_anomalous_recipe_count": len({row.get("recipe_id") for row in anomalies}),
        "unique_anomalous_live_sample_count": len({row.get("live_target_sample_id") for row in anomalies}),
        "recipes_anomalous_in_both_routes": [entry["key"] for entry in by_recipe
                                             if len(entry["routes_in_anomalies"]) > 1],
        "by_recipe": by_recipe, "by_live_sample": by_live_sample, "by_position_mod_8": by_position_mod_8,
        "anomalies_form_contiguous_block": any(block["count"] > 1 for block in blocks),
        "contiguous_blocks": blocks,
        "position_range_of_anomalies": ({"min": min(item["position"] for item in sequence if item["anomaly"]),
                                         "max": max(item["position"] for item in sequence if item["anomaly"])}
                                        if any(item["anomaly"] for item in sequence) else None),
        "timestamp_change_point": "NOT COMPUTABLE -- no per-candidate timestamp is persisted anywhere "
                                  "in CANDIDATE.json (generation_identity or trace)",
        "max_deviation_candidates": max_deviation,
        "requested_vs_achieved_buckets_all": requested_vs_achieved_all,
        "requested_vs_achieved_buckets_anomalies_only": requested_vs_achieved_anomalies,
        "known_q_mismatch_candidates_found_in_anomalies": len(known_mismatches_in_anomalies),
        "known_q_mismatch_candidate_ids": list(KNOWN_Q_AUDIT_MISMATCH_CANDIDATES),
        "c6_bank_lock_join_available": c6_join_available,
        "c6_selected_among_anomalies": (sum(1 for row in anomalies if row.get("c6_selected"))
                                        if c6_join_available else None),
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"available": True, "csv_written": True, "summary_written": True,
           "csv_path": str(anomalies_csv_path), "summary_path": str(summary_path), "summary": summary}


def gpu_anomaly_analysis_command(*, arm: str = "LLM") -> str:
    """TASK L: prepared, never executed here. Read-only on the GPU host."""
    return f"python -m prism_fas.evaluation.c_ext_e6_render --analyze-historical-trace-anomalies --audit-arm {arm}"


# --------------------------------------------------------------------------- #
# TASK A-J (continuation turn): the TWO-GPAT-RENDER-PASS investigation. GPU
# evidence showed the SAME frozen schedule position can carry two GPAT
# candidates with two different `route_binding` (trace["binding"]) values --
# every function below is a pure, read-only transform of population-audit
# rows or a read-only re-inspection of already-frozen C4/C5/C6 lock files.
# None opens an image/prior/model, none touches GPU, none touches target,
# none deletes or mutates any historical candidate.
# --------------------------------------------------------------------------- #

def aggregate_by_route_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK A: population aggregate keyed by (route, route_binding). Discovers
    ALL actual bindings present in the audited rows -- nothing here assumes
    there are exactly two, or that GPAT is the only affected route."""
    from collections import defaultdict

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "error" in row:
            continue
        by_key[(str(row.get("route")), str(row.get("route_binding")))].append(row)

    bindings = []
    for (route, binding), group in sorted(by_key.items()):
        positions = [int(row["position"]) for row in group if row.get("position") is not None]
        bindings.append({
            "route": route, "route_binding": binding,
            "candidate_count": len(group),
            "unique_recipe_count": len({row.get("recipe_id") for row in group}),
            "unique_position_count": len(set(positions)),
            "min_position": min(positions) if positions else None,
            "max_position": max(positions) if positions else None,
            "unique_schedule_key_count": len(set(positions)),
            "candidate_ids_sample": sorted(str(row.get("candidate_id")) for row in group)[:5],
        })
    routes = sorted({route for route, _binding in by_key})
    return {
        "schema_version": "e6-route-binding-population-v1",
        "bindings": bindings,
        "distinct_route_binding_pairs": len(bindings),
        "distinct_bindings_by_route": {
            route: sorted({binding for (r, binding) in by_key if r == route}) for route in routes},
    }


def write_gpat_binding_population_artifacts(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASK A: writes E6_GPAT_BINDING_POPULATION.csv / _SUMMARY.json, additive,
    under this extension's own render/ namespace only. Reuses the SAME
    population-audit rows the other population tools use -- never re-reads a
    candidate file a second time."""
    import csv

    audit = audit_historical_trace_population(repo, arm=arm)
    out_dir = repo / RENDER_DIR
    csv_path = out_dir / "E6_GPAT_BINDING_POPULATION.csv"
    summary_path = out_dir / "E6_GPAT_BINDING_POPULATION_SUMMARY.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audit["available"]:
        summary = {"schema_version": "e6-gpat-binding-population-summary-v1", "available": False,
                  "arm": arm, "reason": audit.get("reason"),
                  "note": "the frozen historical candidate tree is not present on this host; no CSV "
                          "was written, and none was fabricated."}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"available": False, "reason": audit.get("reason"), "csv_written": False,
               "summary_written": True, "csv_path": None, "summary_path": str(summary_path)}

    rows = audit["rows"]
    aggregate = aggregate_by_route_binding(rows)
    fieldnames = list(aggregate["bindings"][0].keys()) if aggregate["bindings"] else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for binding_row in aggregate["bindings"]:
            writer.writerow({**binding_row, "candidate_ids_sample": ";".join(binding_row["candidate_ids_sample"])})

    summary = {"schema_version": "e6-gpat-binding-population-summary-v1", "available": True, "arm": arm,
              "row_count": len(rows), **aggregate,
              "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
              "training_performed": False}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"available": True, "csv_written": True, "summary_written": True,
           "csv_path": str(csv_path), "summary_path": str(summary_path), "aggregate": aggregate}


#: TASK B: the FROZEN_SCHEDULE_KEY, derived strictly from the real
#: `C5_SOURCE_PAIR_PLAN.json` (never invented): that lock's own fields --
#: `positions=2048`, `renders_per_recipe=8` (== EXPECTED_RENDERS_PER_RECIPE),
#: `recipes_per_arm=256`, `route_by_slot` (an 8-entry, slot-indexed, position-
#: INDEPENDENT route table) -- together fix `position = recipe_ordinal *
#: renders_per_recipe + slot`, and route/recipe_ordinal/slot/
#: live_target_sample_id/spoof_source_sample_id are ALL deterministic
#: functions of `position` alone (`c5_source_pair_plan.build_source_pair_plan`/
#: `live_for_position`/`route_for_slot`/`spoof_for_position` take only
#: `position`). `position` is therefore the minimal field uniquely
#: identifying ONE planned render, independent of `generator_binding` -- the
#: one field `candidate_identity()` adds on top of the schedule to select
#: which generator rendered it.
FROZEN_SCHEDULE_KEY_FIELD = "position"


def schedule_key_for_row(row: dict[str, Any]) -> int | None:
    """TASK B: the FROZEN_SCHEDULE_KEY value for one population-audit row."""
    value = row.get(FROZEN_SCHEDULE_KEY_FIELD)
    return int(value) if value is not None else None


def group_by_schedule_key(rows: list[dict[str, Any]], *, route: str) -> dict[int, list[dict[str, Any]]]:
    """TASK B: candidates for one route, grouped by FROZEN_SCHEDULE_KEY."""
    from collections import defaultdict

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "error" in row or row.get("route") != route:
            continue
        key = schedule_key_for_row(row)
        if key is not None:
            grouped[key].append(row)
    return dict(grouped)


def classify_double_gpat_render_pass(grouped_gpat: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """TASK B: key-count distribution + the PROVEN/NOT_PROVEN classification,
    using a rule declared here BEFORE inspecting the actual counts: PROVEN
    only if there are exactly EXPECTED_LLM_GPAT_GENERATED keys, EVERY one with
    EXACTLY 2 candidates (never >2, never a mix, never a different total)."""
    keys_with_1 = sum(1 for group in grouped_gpat.values() if len(group) == 1)
    keys_with_2 = sum(1 for group in grouped_gpat.values() if len(group) == 2)
    keys_with_gt2 = sum(1 for group in grouped_gpat.values() if len(group) > 2)
    total_keys = len(grouped_gpat)
    proven = (total_keys == EXPECTED_LLM_GPAT_GENERATED
              and keys_with_2 == total_keys and keys_with_1 == 0 and keys_with_gt2 == 0)
    return {
        "schema_version": "e6-double-gpat-render-pass-v1",
        "frozen_schedule_key": FROZEN_SCHEDULE_KEY_FIELD,
        "gpat_schedule_keys_total": total_keys,
        "keys_with_1_candidate": keys_with_1,
        "keys_with_2_candidates": keys_with_2,
        "keys_with_gt2_candidates": keys_with_gt2,
        "double_gpat_render_pass": "PROVEN" if proven else "NOT_PROVEN",
    }


#: TASK C: the read-only per-pair comparison fields, in report order.
_PAIR_COMPARISON_FIELDS: tuple[str, ...] = (
    "candidate_id", "route_binding", "recipe_id", "slot", "position",
    "live_target_sample_id", "persisted_recipe_hash", "persisted_graph_hash",
    "trace_requested_coverage", "trace_achieved_coverage",
    "trace_requested_region_pixels", "trace_requested_support_pixels",
    "trace_exact_mask_pixels", "candidate_json_sha256",
)


def pair_gpat_candidates_by_schedule_key(grouped_gpat: dict[int, list[dict[str, Any]]],
                                         *, bank_lock: dict[str, Any] | None = None
                                         ) -> list[dict[str, Any]]:
    """TASK C: for every schedule key with EXACTLY two GPAT candidates, a
    read-only side-by-side comparison row. `bank_lock` (an already-parsed
    C6_BANK_LOCK_LLM.json) is OPTIONAL and only adds q/selected from the
    frozen 'selected' list where resolvable -- never recomputes q."""
    selected_by_id = {str(row["candidate_id"]): row for row in (bank_lock or {}).get("selected") or []}

    pairs = []
    for key in sorted(k for k, group in grouped_gpat.items() if len(group) == 2):
        first, second = sorted(grouped_gpat[key], key=lambda row: str(row.get("candidate_id")))
        pair_row: dict[str, Any] = {"schedule_key": key}
        for field in _PAIR_COMPARISON_FIELDS:
            pair_row[f"{field}_A"] = first.get(field)
            pair_row[f"{field}_B"] = second.get(field)
        for label, row in (("A", first), ("B", second)):
            selected_row = selected_by_id.get(row.get("candidate_id"))
            pair_row[f"c6_selected_{label}"] = selected_row is not None
            pair_row[f"c6_q_{label}"] = selected_row.get("q") if selected_row else None
        pairs.append(pair_row)
    return pairs


def resolve_canonical_gpat_binding(repo: Path) -> dict[str, Any]:
    """TASK D: determine which GPAT binding is CANONICAL using ONLY frozen,
    persisted evidence -- never memory, never chronology-guessing, and never
    this module's own EXPECTED_GPAT_CHECKPOINT_SHA256 alone (that constant is
    CROSS-CHECKED here, not assumed correct).

    Evidence chain, every step read fresh from disk:
      1. `reports/full/c4/GPAT_CONFIG_LOCK.json` -- the frozen C4 scientific-
         search-winning checkpoint (`winning_checkpoint_sha256`).
      2. `reports/full/c5/C5_SYNTHESIS_LOCK.json` -- the ACTIVE (non-
         superseded) C5 lock; its own `gpat_checkpoint_sha256` is the binding
         C5 itself claims produced this candidate pool.
      3. If the active lock carries a `supersedes.archived_lock` pointer,
         THAT EXACT file (never a different file in `superseded/` chosen by
         name pattern or mtime) is read and its `gpat_checkpoint_sha256` +
         `lock_kind`/`usable_as_c6_input` recorded as the SECOND binding.
      4. `reports/full/c6/C6_BANK_LOCK_LLM.json`'s `c5_pool_lock_sha256` -- a
         SHA-256 of the C5 lock FILE BYTES C6 actually consumed, computed
         fresh against the active lock file on disk right now (never trusted
         from a prior run) and compared. This is the strongest evidence: a
         byte-level pin from C6 to one specific C5 lock file.
    A binding is CANONICAL only if EXPECTED_GPAT_CHECKPOINT_SHA256, the C4
    winner (when present), the active C5 lock's own binding, and the C6
    pool-lock SHA-256 pin (when present) all agree. If they disagree, this
    reports AMBIGUOUS rather than guessing.
    """
    proof: list[dict[str, Any]] = []

    c4_path = repo / C4_SCIENTIFIC_LOCK_PATH
    c4_winner = None
    if c4_path.is_file():
        c4_lock = json.loads(c4_path.read_text(encoding="utf-8"))
        c4_winner = c4_lock.get("winning_checkpoint_sha256")
        proof.append({"artifact": C4_SCIENTIFIC_LOCK_PATH, "field": "winning_checkpoint_sha256",
                     "value": c4_winner})

    active_path = repo / C5_SYNTHESIS_LOCK_PATH
    if not active_path.is_file():
        return {"schema_version": "e6-canonical-gpat-binding-v1", "available": False,
               "reason": f"missing {C5_SYNTHESIS_LOCK_PATH} on this host", "proof": proof,
               "canonical_binding_status": "UNAVAILABLE"}
    active_bytes = active_path.read_bytes()
    active_lock = json.loads(active_bytes.decode("utf-8"))
    active_binding = active_lock.get("gpat_checkpoint_sha256")
    active_sha256 = hashlib.sha256(active_bytes).hexdigest()
    proof.append({"artifact": C5_SYNTHESIS_LOCK_PATH, "field": "gpat_checkpoint_sha256",
                 "value": active_binding, "lock_kind": active_lock.get("lock_kind"),
                 "is_scientific_lock": active_lock.get("is_scientific_lock"),
                 "file_sha256": active_sha256})

    supersedes = active_lock.get("supersedes") or {}
    archived_lock_relpath = supersedes.get("archived_lock")
    second_binding = None
    second_role = "UNKNOWN"
    if archived_lock_relpath:
        archived_path = repo / archived_lock_relpath
        if archived_path.is_file():
            archived_bytes = archived_path.read_bytes()
            archived_lock = json.loads(archived_bytes.decode("utf-8"))
            second_binding = archived_lock.get("gpat_checkpoint_sha256")
            second_role = ("SUPERSEDED"
                          if archived_lock.get("lock_kind") == "terminal_audit_record"
                          and archived_lock.get("usable_as_c6_input") is False
                          else "UNKNOWN")
            proof.append({"artifact": archived_lock_relpath, "field": "gpat_checkpoint_sha256",
                         "value": second_binding, "lock_kind": archived_lock.get("lock_kind"),
                         "usable_as_c6_input": archived_lock.get("usable_as_c6_input"),
                         "why_not_usable": archived_lock.get("why_not_usable"),
                         "file_sha256": hashlib.sha256(archived_bytes).hexdigest(),
                         "referenced_by": f"{C5_SYNTHESIS_LOCK_PATH}#supersedes.archived_lock"})

    c6_path = repo / C6_BANK_LOCK_LLM_PATH
    c6_pins_active = None
    c6_pool_lock_sha256 = None
    if c6_path.is_file():
        c6_lock = json.loads(c6_path.read_text(encoding="utf-8"))
        c6_pool_lock_sha256 = c6_lock.get("c5_pool_lock_sha256")
        c6_pins_active = (c6_pool_lock_sha256 == active_sha256)
        proof.append({"artifact": C6_BANK_LOCK_LLM_PATH, "field": "c5_pool_lock_sha256",
                     "value": c6_pool_lock_sha256, "matches_active_c5_lock_file_sha256": c6_pins_active})

    agree = (active_binding is not None
             and (c4_winner is None or c4_winner == active_binding)
             and active_binding == EXPECTED_GPAT_CHECKPOINT_SHA256
             and (c6_pins_active is not False))
    canonical_binding = active_binding if agree else None

    return {
        "schema_version": "e6-canonical-gpat-binding-v1", "available": True,
        "frozen_expected_gpat_binding": EXPECTED_GPAT_CHECKPOINT_SHA256,
        "c4_winning_checkpoint_sha256": c4_winner,
        "active_c5_lock_gpat_checkpoint_sha256": active_binding,
        "second_gpat_binding": second_binding,
        "second_gpat_population_role": second_role,
        "c6_bank_lock_pins_active_c5_lock": c6_pins_active,
        "canonical_gpat_binding": canonical_binding,
        "canonical_binding_status": "PROVEN" if canonical_binding else "AMBIGUOUS",
        "proof": proof,
    }


def candidate_id_depends_on_route_binding() -> dict[str, Any]:
    """TASK E: mechanical (not inferential) answer, from
    `c5_source_pair_plan.candidate_identity`'s own hash material.

    `candidate_identity()` hashes, in order: SCHEMA_VERSION,
    source_pair_plan_identity, arm, recipe_bank_identity, recipe_id,
    recipe_ordinal, slot, position, route, live_target_sample_id,
    spoof_source_sample_id, package_identity, ontology_identity,
    **generator_binding**, seed. `generator_binding` is the C4 winning GPAT
    checkpoint SHA-256 on the GPAT route (per that function's own docstring).
    Because it is direct hash material, two renders of the SAME schedule
    position under two different `generator_binding` values are MECHANICALLY
    GUARANTEED two different candidate_ids -- this is not a bug, a collision,
    or a violation of any uniqueness guarantee; it is the intended behavior of
    a content-addressed id that binds the generator that actually produced
    the pixels.
    """
    return {
        "schema_version": "e6-candidate-id-binding-dependency-v1",
        "candidate_id_depends_on_route_binding": True,
        "hash_material_fields": [
            "SCHEMA_VERSION", "source_pair_plan_identity", "arm", "recipe_bank_identity",
            "recipe_id", "recipe_ordinal", "slot", "position", "route",
            "live_target_sample_id", "spoof_source_sample_id", "package_identity",
            "ontology_identity", "generator_binding", "seed",
        ],
        "source": "prism_fas.synthesis.c5_source_pair_plan.candidate_identity",
        "mechanism": "generator_binding (GPAT checkpoint SHA-256 on the gpat route, "
                    "PhysicsEngine version on the physics route) is direct SHA-256 hash "
                    "material of candidate_identity(); it is not recomputed or derived from "
                    "anything else. Two renders of the same schedule position under two "
                    "different generator_binding values therefore produce two genuinely "
                    "distinct, non-colliding candidate_ids by construction.",
    }


def explain_second_population_reachability() -> dict[str, Any]:
    """TASK F support: WHY the non-canonical GPAT population is never read by
    any downstream stage, proven from the actual record-collection code
    rather than inferred. `synthesis.c5_render.collect_records` (C5's own
    record collector, reused by the finalize/verify call sites both C5 and C6
    share) iterates `plan["candidates"]` and reads exactly
    `candidate_dir(work_root, arm, row["candidate_id"]) / CANDIDATE.json` for
    each PLANNED row -- it never lists the candidate_root directory. Because
    `row["candidate_id"]` is computed from the CURRENT `gpat_checkpoint_sha256`
    (the one the freshly-verified C4 lock names right now --
    `pipeline.adapters.c5.reconstruct_current_c5_inputs`), a superseded
    binding's candidate_id values are never computed and its on-disk
    directories are never addressed by any real call path."""
    return {
        "schema_version": "e6-second-population-reachability-v1",
        "collector": "prism_fas.synthesis.c5_render.collect_records",
        "enumeration_mechanism": "iterates plan['candidates'] and reads "
                                 "candidate_dir(work_root, arm, row['candidate_id']) / CANDIDATE.json "
                                 "for each PLANNED row; never globs/scans candidate_root",
        "candidate_id_source": "row['candidate_id'] is computed by the arm-plan builder using the "
                               "CURRENT verified C4 checkpoint as generator_binding "
                               "(pipeline.adapters.c5.reconstruct_current_c5_inputs)",
        "conclusion": "a superseded binding's candidate_ids are never computed by any real call "
                     "path, so its on-disk CANDIDATE.json directories are structurally "
                     "unreachable/inert to C5 finalize/verify and C6 matched-bank building, not "
                     "merely unused by convention",
    }


def build_canonical_population_view(rows: list[dict[str, Any]], *, canonical_gpat_binding: str) -> dict[str, Any]:
    """TASK G: a READ-ONLY canonical population VIEW -- filters GPAT rows down
    to `canonical_gpat_binding`, keeps EVERY physics row untouched (physics
    has only ever shown one `route_binding` -- EXPECTED_PHYSICS_ENGINE_VERSION
    -- across every lock TASK D inspects), and NEVER deletes or mutates the
    input `rows` list. Only meaningful once TASK D has already proven a
    single canonical binding; callers must check that themselves."""
    canonical_rows = [row for row in rows if "error" not in row
                      and (row.get("route") != "gpat" or row.get("route_binding") == canonical_gpat_binding)]
    excluded_rows = [row for row in rows if "error" not in row
                     and row.get("route") == "gpat" and row.get("route_binding") != canonical_gpat_binding]
    gpat_n = sum(1 for row in canonical_rows if row.get("route") == "gpat")
    physics_n = sum(1 for row in canonical_rows if row.get("route") == "physics")
    return {
        "schema_version": "e6-canonical-population-view-v1",
        "unfiltered_tree_view_note": "the earlier, UNFILTERED population audit "
                                     "(E6_HISTORICAL_TRACE_POPULATION.csv/_SUMMARY.json) is untouched "
                                     "and remains the UNFILTERED_HISTORICAL_TREE_VIEW; this is a NEW, "
                                     "additive, read-only view alongside it, never a replacement",
        "canonical_gpat_binding": canonical_gpat_binding,
        "canonical_population_total": len(canonical_rows),
        "canonical_gpat_n": gpat_n,
        "canonical_physics_n": physics_n,
        "excluded_non_canonical_gpat_count": len(excluded_rows),
        "rows": canonical_rows,
        "excluded_rows_candidate_ids": [row.get("candidate_id") for row in excluded_rows],
    }


def write_canonical_population_artifacts(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASK G: writes E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv /
    _SUMMARY.json, ONLY when TASK D's `resolve_canonical_gpat_binding` proves
    a single canonical binding. NEVER overwrites
    E6_HISTORICAL_TRACE_POPULATION.csv (that file is explicitly marked
    UNFILTERED_HISTORICAL_TREE_VIEW in the summary this writes, not edited or
    deleted)."""
    import csv

    audit = audit_historical_trace_population(repo, arm=arm)
    out_dir = repo / RENDER_DIR
    csv_path = out_dir / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv"
    summary_path = out_dir / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION_SUMMARY.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audit["available"]:
        summary = {"schema_version": "e6-canonical-population-summary-v1", "available": False, "arm": arm,
                  "reason": audit.get("reason"), "note": "population tree not present on this host; no "
                          "CSV was written, and none was fabricated."}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"available": False, "reason": audit.get("reason"), "csv_written": False,
               "summary_written": True, "csv_path": None, "summary_path": str(summary_path)}

    resolution = resolve_canonical_gpat_binding(repo)
    if resolution.get("canonical_binding_status") != "PROVEN":
        summary = {"schema_version": "e6-canonical-population-summary-v1", "available": True, "arm": arm,
                  "canonical_binding_resolution": resolution,
                  "note": "TASK D did not prove a single canonical binding on this host; refusing to "
                          "build a canonical view rather than guess which population is scientific"}
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return {"available": True, "csv_written": False, "summary_written": True,
               "csv_path": None, "summary_path": str(summary_path),
               "canonical_binding_resolution": resolution}

    canonical_binding = resolution["canonical_gpat_binding"]
    view = build_canonical_population_view(audit["rows"], canonical_gpat_binding=canonical_binding)
    integrity = population_integrity_check(view["rows"])
    anomalies = filter_anomalies(view["rows"])

    fieldnames = list(view["rows"][0].keys()) if view["rows"] else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in view["rows"]:
            writer.writerow(row)

    summary = {
        "schema_version": "e6-canonical-population-summary-v1", "available": True, "arm": arm,
        "canonical_binding_resolution": resolution,
        "canonical_population_total": view["canonical_population_total"],
        "canonical_gpat_n": view["canonical_gpat_n"],
        "canonical_physics_n": view["canonical_physics_n"],
        "canonical_anomaly_count": len(anomalies),
        "canonical_gpat_anomaly_count": sum(1 for row in anomalies if row.get("route") == "gpat"),
        "canonical_physics_anomaly_count": sum(1 for row in anomalies if row.get("route") == "physics"),
        "excluded_non_canonical_gpat_count": view["excluded_non_canonical_gpat_count"],
        "integrity": integrity,
        "unfiltered_tree_view_note": view["unfiltered_tree_view_note"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"available": True, "csv_written": True, "summary_written": True,
           "csv_path": str(csv_path), "summary_path": str(summary_path), "summary": summary}


def classify_known_q_mismatch_bindings(rows: list[dict[str, Any]], *,
                                       canonical_gpat_binding: str | None) -> dict[str, Any]:
    """TASK H: for the 3 known q-mismatch candidates, reports their
    route_binding and whether it is canonical -- computed from the audited
    rows, never assumed."""
    by_id = {row.get("candidate_id"): row for row in rows if "error" not in row}
    results = []
    for candidate_id in KNOWN_Q_AUDIT_MISMATCH_CANDIDATES:
        row = by_id.get(candidate_id)
        binding = row.get("route_binding") if row else None
        results.append({
            "candidate_id": candidate_id,
            "found": row is not None,
            "route": row.get("route") if row else None,
            "route_binding": binding,
            "is_canonical_binding": (binding == canonical_gpat_binding
                                     if binding is not None and canonical_gpat_binding is not None else None),
        })
    known = [entry for entry in results if entry["found"]]
    all_canonical = bool(known) and all(entry["is_canonical_binding"] is True for entry in known)
    any_non_canonical = any(entry["is_canonical_binding"] is False for entry in known)
    return {
        "schema_version": "e6-known-q-mismatch-binding-classification-v1",
        "candidates": results,
        "three_known_q_mismatch_binding": ("CANONICAL" if all_canonical
                                           else "NON_CANONICAL" if any_non_canonical
                                           else "MIXED_OR_UNRESOLVED"),
        "three_known_q_mismatch_canonical": all_canonical,
        "explains_historical_q_blocker": (False if all_canonical
                                          else (True if any_non_canonical and not all_canonical else None)),
    }


def run_gpat_binding_investigation(repo: Path, *, arm: str = "LLM") -> dict[str, Any]:
    """TASKS A-H, orchestrated: one read-only pass that answers the whole
    TWO-GPAT-RENDER-PASS question and writes every additive artifact this
    continuation turn asked for. Never renders, never trains, never touches
    target, never calls an LLM, never deletes or mutates a historical
    candidate or lock."""
    audit = audit_historical_trace_population(repo, arm=arm)
    binding_population = write_gpat_binding_population_artifacts(repo, arm=arm)

    # TASKS D/E/F read only frozen C4/C5/C6 LOCK files, never the candidate
    # tree -- they are resolved regardless of whether the population audit
    # itself is available on this host (e.g. a laptop with the frozen locks
    # but not `runs/full/c5/scientific/candidates`).
    canonical_resolution = resolve_canonical_gpat_binding(repo)
    candidate_id_dependency = candidate_id_depends_on_route_binding()
    reachability = explain_second_population_reachability()

    if not audit["available"]:
        summary = {
            "schema_version": "e6-gpat-binding-investigation-v1", "available": False,
            "reason": audit.get("reason"), "binding_population": binding_population,
            "canonical_resolution": canonical_resolution,
            "candidate_id_dependency": candidate_id_dependency,
            "second_population_reachability": reachability,
            "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
            "training_performed": False,
        }
        out_dir = repo / RENDER_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        investigation_summary_path = out_dir / "E6_GPAT_BINDING_INVESTIGATION_SUMMARY.json"
        investigation_summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return {**summary, "summary_path": str(investigation_summary_path)}

    rows = audit["rows"]
    grouped_gpat = group_by_schedule_key(rows, route="gpat")
    schedule_key_classification = classify_double_gpat_render_pass(grouped_gpat)

    bank_lock_path = repo / C6_BANK_LOCK_LLM_PATH
    bank_lock = (json.loads(bank_lock_path.read_text(encoding="utf-8"))
                if bank_lock_path.is_file() else None)
    pairs = pair_gpat_candidates_by_schedule_key(grouped_gpat, bank_lock=bank_lock)

    canonical_binding = (canonical_resolution.get("canonical_gpat_binding")
                         if canonical_resolution.get("canonical_binding_status") == "PROVEN" else None)

    canonical_view = (build_canonical_population_view(rows, canonical_gpat_binding=canonical_binding)
                      if canonical_binding else None)
    canonical_artifacts = (write_canonical_population_artifacts(repo, arm=arm) if canonical_binding else None)

    known_mismatch_classification = classify_known_q_mismatch_bindings(
        rows, canonical_gpat_binding=canonical_binding)

    summary = {
        "schema_version": "e6-gpat-binding-investigation-v1", "available": True, "arm": arm,
        "unfiltered_tree_total": len([row for row in rows if "error" not in row]),
        "expected_scientific_total": EXPECTED_LLM_TOTAL_GENERATED,
        "binding_population": {key: value for key, value in binding_population.items()
                               if key not in ("aggregate",)},
        "gpat_bindings": binding_population.get("aggregate", {}).get(
            "distinct_bindings_by_route", {}).get("gpat", []),
        "schedule_key_classification": schedule_key_classification,
        "pair_count": len(pairs),
        "canonical_resolution": canonical_resolution,
        "candidate_id_dependency": candidate_id_dependency,
        "second_population_reachability": reachability,
        "canonical_population_total": (canonical_view["canonical_population_total"] if canonical_view else None),
        "canonical_gpat_n": (canonical_view["canonical_gpat_n"] if canonical_view else None),
        "canonical_physics_n": (canonical_view["canonical_physics_n"] if canonical_view else None),
        "known_q_mismatch_classification": known_mismatch_classification,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }
    out_dir = repo / RENDER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "E6_GPAT_SCHEDULE_KEY_PAIRS.json"
    pairs_path.write_text(json.dumps({"schema_version": "e6-gpat-schedule-key-pairs-v1", "pairs": pairs},
                                    indent=2, default=str), encoding="utf-8")
    investigation_summary_path = out_dir / "E6_GPAT_BINDING_INVESTIGATION_SUMMARY.json"
    investigation_summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {**summary, "pairs_path": str(pairs_path), "summary_path": str(investigation_summary_path),
           "canonical_artifacts": canonical_artifacts}


def gpu_gpat_binding_investigation_command(*, arm: str = "LLM") -> str:
    """TASKS A-H: prepared, never executed here. Read-only on the GPU host."""
    return f"python -m prism_fas.evaluation.c_ext_e6_render --investigate-gpat-binding --audit-arm {arm}"


# --------------------------------------------------------------------------- #
# TASK A-J (continuation turn): the CANONICAL 42-anomaly characterization --
# operates STRICTLY on E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv (the
# GPU-produced canonical view from the double-GPAT-render-pass investigation),
# never on the 3058-row UNFILTERED_HISTORICAL_TREE_VIEW except read-only for
# the cross-binding comparison TASK F asks for. Every function is a pure,
# read-only transform or a pure recompilation of a frozen recipe (no image,
# no model, no GPU, no target). None writes anywhere but this extension's own
# render/ namespace; none mutates the canonical CSV or any historical lock.
# --------------------------------------------------------------------------- #

#: Chosen BEFORE inspecting any real field-match distribution -- a coarser
#: tolerance than TRACE_COMPARISON_TOLERANCE (1e-6) because a recipe/graph
#: field is compared to a coverage value that may itself be a rounded
#: display value; still tight enough that a coincidental match is unlikely.
FIELD_MATCH_TOLERANCE = 1e-4

#: The 3 known q-mismatch candidates' recipe_ids, read directly from the real,
#: frozen reports/full/c6/C6_BANK_LOCK_LLM.json 'selected' rows earlier this
#: investigation (never guessed) -- R-000221/c5syn_039..., R-000354/
#: c5syn_057..., R-000167/c5syn_058....
KNOWN_Q_MISMATCH_RECIPE_MAP: dict[str, str] = {
    "c5syn_0390812685e6403952baeb67": "R-000221",
    "c5syn_057aab8fef90ada42997a1a4": "R-000354",
    "c5syn_0588f57b2499484387d2b7af": "R-000167",
}


def _coerce_canonical_csv_row(row: dict[str, str]) -> dict[str, Any]:
    """CSV round-trips every value through str(); this restores the exact
    Python types `audit_historical_trace_population`'s rows originally had,
    never guessing a type for a field it doesn't recognize."""
    def _bool_or_none(value: Any) -> Any:
        if value == "True":
            return True
        if value == "False":
            return False
        if value in ("None", ""):
            return None
        return value

    def _float_or_none(value: Any) -> float | None:
        if value in ("None", "", None):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int_or_none(value: Any) -> int | None:
        if value in ("None", "", None):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    bool_fields = ("recipe_hash_matches", "graph_hash_matches", "bank_identity_matches",
                  "requested_equals_recipe_coverage", "requested_equals_achieved_coverage",
                  "requested_equals_support_ratio", "achieved_equals_support_ratio")
    float_fields = ("recipe_geometry_coverage", "trace_requested_coverage", "trace_achieved_coverage",
                    "support_over_region")
    int_fields = ("recipe_ordinal", "slot", "position", "trace_requested_region_pixels",
                 "trace_requested_support_pixels", "trace_exact_mask_pixels")
    string_fields = ("candidate_id", "route", "recipe_id", "live_target_sample_id", "route_binding",
                     "persisted_recipe_hash", "current_recipe_hash", "persisted_graph_hash",
                     "current_graph_hash", "persisted_recipe_bank_identity", "current_recipe_bank_identity",
                     "candidate_json_sha256")

    coerced: dict[str, Any] = dict(row)
    for field in bool_fields:
        if field in coerced:
            coerced[field] = _bool_or_none(coerced[field])
    for field in float_fields:
        if field in coerced:
            coerced[field] = _float_or_none(coerced[field])
    for field in int_fields:
        if field in coerced:
            coerced[field] = _int_or_none(coerced[field])
    for field in string_fields:
        if field in coerced and coerced[field] in ("None", ""):
            coerced[field] = None
    return coerced


def load_canonical_population_csv(repo: Path) -> dict[str, Any]:
    """TASK A: loads E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv, strictly
    read-only, with the SAME row shape `audit_historical_trace_population`
    produces (this file's own writer, `write_canonical_population_artifacts`,
    writes exactly those rows) -- type-coerced back from CSV strings. Never
    fabricates a population when the file is absent on this host."""
    import csv

    path = repo / RENDER_DIR / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv"
    if not path.is_file():
        return {"schema_version": "e6-canonical-population-csv-load-v1", "available": False,
               "reason": f"missing {path.as_posix()} on this host -- run "
                        "write_canonical_population_artifacts (via --investigate-gpat-binding) on the "
                        "GPU host first", "rows": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [_coerce_canonical_csv_row(row) for row in reader]
    return {"schema_version": "e6-canonical-population-csv-load-v1", "available": True,
           "path": str(path), "row_count": len(rows), "rows": rows}


def summarize_canonical_recipe_groups(rows: list[dict[str, Any]],
                                      anomaly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK A/B: per-recipe canonical statistics -- total renders, anomalous
    renders, per-route counts, recipe.geometry.coverage, the DISTINCT
    trace.requested_coverage values seen for that recipe, and achieved-
    coverage range. `is_anomalous_recipe`/`all_renders_anomalous` are the raw
    material TASK B's ANOMALY_DETERMINED_BY_RECIPE_ID classification uses."""
    from collections import defaultdict

    by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "error" in row:
            continue
        by_recipe[row.get("recipe_id")].append(row)
    anomaly_ids = {row.get("candidate_id") for row in anomaly_rows}

    groups = []
    for recipe_id, recipe_rows in sorted(by_recipe.items()):
        anomalous = [row for row in recipe_rows if row.get("candidate_id") in anomaly_ids]
        coverages = sorted({row["trace_requested_coverage"] for row in recipe_rows
                           if row.get("trace_requested_coverage") is not None})
        achieved = [row["trace_achieved_coverage"] for row in recipe_rows
                   if row.get("trace_achieved_coverage") is not None]
        geometry_coverage = next((row.get("recipe_geometry_coverage") for row in recipe_rows
                                 if row.get("recipe_geometry_coverage") is not None), None)
        groups.append({
            "recipe_id": recipe_id,
            "total_canonical_renders": len(recipe_rows),
            "anomalous_renders": len(anomalous),
            "gpat_renders": sum(1 for row in recipe_rows if row.get("route") == "gpat"),
            "physics_renders": sum(1 for row in recipe_rows if row.get("route") == "physics"),
            "recipe_geometry_coverage": geometry_coverage,
            "unique_trace_requested_coverage_values": coverages,
            "requested_coverage_constant_within_recipe": len(coverages) <= 1,
            "min_achieved_coverage": min(achieved) if achieved else None,
            "max_achieved_coverage": max(achieved) if achieved else None,
            "is_anomalous_recipe": len(anomalous) > 0,
            "all_renders_anomalous": len(anomalous) == len(recipe_rows) and len(recipe_rows) > 0,
        })
    return groups


def classify_anomaly_determined_by_recipe(groups: list[dict[str, Any]]) -> str:
    """TASK B: TRUE only if EVERY anomalous recipe has every one of its
    canonical renders anomalous; FALSE if none do; PARTIAL for a genuine
    mix -- declared before inspecting the real GPU distribution."""
    anomalous_groups = [group for group in groups if group["is_anomalous_recipe"]]
    if not anomalous_groups:
        return "FALSE"
    all_true = all(group["all_renders_anomalous"] for group in anomalous_groups)
    any_true = any(group["all_renders_anomalous"] for group in anomalous_groups)
    if all_true:
        return "TRUE"
    if any_true:
        return "PARTIAL"
    return "FALSE"


def load_frozen_recipe_json(repo: Path, recipe_id: str) -> dict[str, Any] | None:
    """TASK C: the RAW, frozen recipe JSON payload (never the parsed pydantic
    object) from the real `assets/recipe_banks/c3/llm/recipes.jsonl` -- the
    exact bytes a recipe-schema-agnostic recursive flattener can walk."""
    path = repo / RECIPE_BANK_LLM_JSONL_PATH
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("recipe_id") == recipe_id:
            return payload
    return None


def flatten_scalar_fields(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """TASK C/D: recursively flattens ANY nested dict/list/tuple payload down
    to `{dotted.or[indexed].path: scalar}` for every int/float/bool leaf --
    schema-agnostic on purpose, so a NEW recipe/graph field added later is
    still found without editing this function."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten_scalar_fields(value, prefix=f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            flat.update(flatten_scalar_fields(value, prefix=f"{prefix}[{index}]"))
    elif isinstance(obj, bool):
        flat[prefix] = obj
    elif isinstance(obj, (int, float)):
        flat[prefix] = obj
    return flat


def match_scalar_fields_to_value(flat: dict[str, Any], target: float | None, *,
                                 tolerance: float = FIELD_MATCH_TOLERANCE) -> dict[str, list[str]]:
    """TASK C/D: exact and fixed-tolerance field names matching `target` --
    booleans are never compared to a coverage float, and a None target
    (non-constant requested_coverage within the recipe) yields no matches
    rather than a spurious one."""
    exact: list[str] = []
    within_tolerance: list[str] = []
    if target is None:
        return {"matching_fields_exact": exact, "matching_fields_tolerance": within_tolerance}
    for field, value in flat.items():
        if isinstance(value, bool):
            continue
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if fvalue == target:
            exact.append(field)
        elif abs(fvalue - target) <= tolerance:
            within_tolerance.append(field)
    return {"matching_fields_exact": sorted(exact), "matching_fields_tolerance": sorted(within_tolerance)}


def compare_anomalous_recipe_fields(repo: Path, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK C: for every anomalous recipe, flattens the frozen RAW recipe
    JSON and compares every numeric scalar against that recipe's
    trace.requested_coverage (only when it is constant within the recipe --
    TASK B's own finding -- otherwise reports None rather than picking one
    arbitrarily)."""
    results = []
    for group in groups:
        if not group["is_anomalous_recipe"]:
            continue
        recipe_id = group["recipe_id"]
        target = (group["unique_trace_requested_coverage_values"][0]
                 if group["requested_coverage_constant_within_recipe"]
                 and group["unique_trace_requested_coverage_values"] else None)
        payload = load_frozen_recipe_json(repo, recipe_id)
        if payload is None:
            results.append({"recipe_id": recipe_id, "recipe_geometry_coverage": group["recipe_geometry_coverage"],
                            "historical_trace_requested_coverage": target,
                            "matching_recipe_fields_exact": [], "matching_recipe_fields_tolerance": [],
                            "reason": "frozen recipe JSON not found in the local recipe bank"})
            continue
        matches = match_scalar_fields_to_value(flatten_scalar_fields(payload), target)
        results.append({
            "recipe_id": recipe_id, "recipe_geometry_coverage": group["recipe_geometry_coverage"],
            "historical_trace_requested_coverage": target,
            "matching_recipe_fields_exact": matches["matching_fields_exact"],
            "matching_recipe_fields_tolerance": matches["matching_fields_tolerance"],
        })
    return results


def compare_anomalous_compiled_graph_fields(repo: Path, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK D: for every anomalous recipe, PURELY recompiles the frozen
    recipe (compile_recipe -- no image, no model, no GPU) and compares every
    numeric scalar in the full compiled graph payload (region_mask_policy,
    every node's parameters, capture, conditioning) against
    trace.requested_coverage."""
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    anomalous = [group for group in groups if group["is_anomalous_recipe"]]
    ontology_path = repo / ONTOLOGY_CONFIG_PATH
    if not ontology_path.is_file():
        return [{"recipe_id": group["recipe_id"], "matching_graph_fields_exact": [],
                "matching_graph_fields_tolerance": [], "reason": "ontology not present on this host"}
               for group in anomalous]
    ontology = load_ontology(ontology_path)

    results = []
    for group in anomalous:
        recipe_id = group["recipe_id"]
        target = (group["unique_trace_requested_coverage_values"][0]
                 if group["requested_coverage_constant_within_recipe"]
                 and group["unique_trace_requested_coverage_values"] else None)
        payload = load_frozen_recipe_json(repo, recipe_id)
        if payload is None:
            results.append({"recipe_id": recipe_id, "matching_graph_fields_exact": [],
                            "matching_graph_fields_tolerance": [],
                            "reason": "frozen recipe JSON not found in the local recipe bank"})
            continue
        graph = compile_recipe(parse_recipe(payload), ontology, bank_id="c3_llm")
        matches = match_scalar_fields_to_value(flatten_scalar_fields(graph.payload(include_hash=False)), target)
        results.append({
            "recipe_id": recipe_id, "graph_hash": graph.graph_hash,
            "historical_trace_requested_coverage": target,
            "matching_graph_fields_exact": matches["matching_fields_exact"],
            "matching_graph_fields_tolerance": matches["matching_fields_tolerance"],
        })
    return results


def classify_alternate_coverage_source(recipe_matches: list[dict[str, Any]],
                                       graph_matches: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK D/E: derives ALTERNATE_COVERAGE_SOURCE strictly from TASK C/D's
    own exact-match results. CODE_JUSTIFIED_TRANSFORMATION is deliberately
    NEVER auto-assigned here -- citing an actual source-code mapping is a
    manual source-reading step (TASK E's own instruction: 'DO NOT fit
    arbitrary formulas'), not something a numeric-coincidence check may
    assert. A tolerance-only (non-exact) numeric match is reported as a
    candidate for manual follow-up, never as a finding."""
    any_recipe_exact = any(match.get("matching_recipe_fields_exact") for match in recipe_matches)
    any_graph_exact = any(match.get("matching_graph_fields_exact") for match in graph_matches)
    any_recipe_tolerance = any(match.get("matching_recipe_fields_tolerance") for match in recipe_matches)
    any_graph_tolerance = any(match.get("matching_graph_fields_tolerance") for match in graph_matches)

    if any_recipe_exact:
        source = "EXACT_RECIPE_FIELD"
    elif any_graph_exact:
        source = "EXACT_GRAPH_FIELD"
    else:
        source = "NONE_FOUND"
    return {
        "alternate_value_present_in_recipe": any_recipe_exact,
        "alternate_value_present_in_compiled_graph": any_graph_exact,
        "alternate_coverage_source": source,
        "tolerance_only_candidates_for_manual_code_review": {
            "recipe_fields": any_recipe_tolerance, "graph_fields": any_graph_tolerance},
        "note": "CODE_JUSTIFIED_TRANSFORMATION is never auto-assigned by this function -- it requires "
               "citing an actual source-code mapping, a manual step. A tolerance-only numeric match is "
               "surfaced for manual review, never asserted as the source.",
    }


def cross_route_and_binding_consistency(canonical_rows: list[dict[str, Any]],
                                        unfiltered_rows: list[dict[str, Any]], *,
                                        canonical_gpat_binding: str | None,
                                        second_gpat_binding: str | None,
                                        anomalous_recipe_ids: set[str]) -> dict[str, Any]:
    """TASK F: for every anomalous recipe, compares trace.requested_coverage
    across (a) canonical GPAT vs Physics (both from the CANONICAL rows) and
    (b) canonical GPAT vs the SUPERSEDED GPAT binding (read-only, from the
    UNFILTERED audit, filtered to `second_gpat_binding` -- superseded rows
    are used ONLY for this read-only cross-check, never folded into any
    scientific count)."""
    from collections import defaultdict

    canonical_by_recipe_route: dict[str, dict[str, set[float]]] = defaultdict(lambda: defaultdict(set))
    for row in canonical_rows:
        if "error" in row or row.get("recipe_id") not in anomalous_recipe_ids:
            continue
        value = row.get("trace_requested_coverage")
        if value is not None:
            canonical_by_recipe_route[row["recipe_id"]][row.get("route")].add(value)

    superseded_by_recipe: dict[str, set[float]] = defaultdict(set)
    for row in unfiltered_rows:
        if ("error" in row or row.get("route") != "gpat" or second_gpat_binding is None
                or row.get("route_binding") != second_gpat_binding
                or row.get("recipe_id") not in anomalous_recipe_ids):
            continue
        value = row.get("trace_requested_coverage")
        if value is not None:
            superseded_by_recipe[row["recipe_id"]].add(value)

    per_recipe = []
    comparable_route_pairs = disagreeing_route_pairs = 0
    comparable_binding_pairs = disagreeing_binding_pairs = 0
    for recipe_id in sorted(anomalous_recipe_ids):
        routes = canonical_by_recipe_route.get(recipe_id, {})
        gpat_values = routes.get("gpat", set())
        physics_values = routes.get("physics", set())
        superseded_values = superseded_by_recipe.get(recipe_id, set())

        same_across_routes = None
        if gpat_values and physics_values:
            comparable_route_pairs += 1
            same_across_routes = gpat_values == physics_values
            if not same_across_routes:
                disagreeing_route_pairs += 1

        same_across_bindings = None
        if gpat_values and superseded_values:
            comparable_binding_pairs += 1
            same_across_bindings = gpat_values == superseded_values
            if not same_across_bindings:
                disagreeing_binding_pairs += 1

        per_recipe.append({
            "recipe_id": recipe_id, "canonical_gpat_values": sorted(gpat_values),
            "physics_values": sorted(physics_values), "superseded_gpat_values": sorted(superseded_values),
            "same_across_routes": same_across_routes, "same_across_bindings": same_across_bindings,
        })

    # None means "no comparable data" and must never collapse into False --
    # the same rule TRACE_COMPARISON_TOLERANCE comparisons use elsewhere in
    # this module (a missing value is 'not comparable', never a mismatch).
    same_across_routes_overall = (None if comparable_route_pairs == 0
                                  else disagreeing_route_pairs == 0)
    same_across_bindings_overall = (None if comparable_binding_pairs == 0
                                    else disagreeing_binding_pairs == 0)

    return {
        "per_recipe": per_recipe,
        "comparable_route_pairs": comparable_route_pairs,
        "comparable_binding_pairs": comparable_binding_pairs,
        "same_recipe_same_alternate_coverage_across_routes": same_across_routes_overall,
        "same_recipe_same_alternate_coverage_across_gpat_bindings": same_across_bindings_overall,
    }


def select_normal_control_recipes(rows: list[dict[str, Any]], *, anomalous_recipe_ids: set[str],
                                  max_controls_per_recipe: int = 1) -> list[dict[str, Any]]:
    """TASK G: for each anomalous recipe, the nearest-geometry-coverage
    NORMAL recipe (every one of ITS canonical renders satisfies
    requested_equals_recipe_coverage==True) -- confirms the anomaly is not
    an artifact of this analysis being overly broad. Never claims
    significance, only reports the control's own pass/fail."""
    from collections import defaultdict

    by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "error" in row:
            continue
        by_recipe[row.get("recipe_id")].append(row)

    anomalous_coverage = {}
    for recipe_id in anomalous_recipe_ids:
        coverage = next((row.get("recipe_geometry_coverage") for row in by_recipe.get(recipe_id, [])
                        if row.get("recipe_geometry_coverage") is not None), None)
        if coverage is not None:
            anomalous_coverage[recipe_id] = coverage

    normal_recipe_ids = [recipe_id for recipe_id, recipe_rows in by_recipe.items()
                        if recipe_id not in anomalous_recipe_ids
                        and all(row.get("requested_equals_recipe_coverage") is True for row in recipe_rows)]

    controls = []
    for anomalous_id, coverage in sorted(anomalous_coverage.items()):
        scored = []
        for normal_id in normal_recipe_ids:
            normal_rows = by_recipe[normal_id]
            normal_coverage = next((row.get("recipe_geometry_coverage") for row in normal_rows
                                   if row.get("recipe_geometry_coverage") is not None), None)
            if normal_coverage is not None:
                scored.append((abs(normal_coverage - coverage), normal_id, normal_rows, normal_coverage))
        scored.sort(key=lambda item: item[0])
        for _distance, normal_id, normal_rows, normal_coverage in scored[:max_controls_per_recipe]:
            controls.append({
                "anomalous_recipe_id": anomalous_id, "control_recipe_id": normal_id,
                "control_geometry_coverage": normal_coverage,
                "control_all_requested_equals_recipe_coverage": all(
                    row.get("requested_equals_recipe_coverage") is True for row in normal_rows),
            })
    return controls


def relate_known_q_mismatches_to_recipe_class(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK H: relates the 3 known historical-q-mismatch candidates to their
    recipe's TASK B anomaly classification. Never recomputes q."""
    by_recipe = {group["recipe_id"]: group for group in groups}
    results = []
    for candidate_id, recipe_id in KNOWN_Q_MISMATCH_RECIPE_MAP.items():
        group = by_recipe.get(recipe_id)
        results.append({
            "candidate_id": candidate_id, "recipe_id": recipe_id,
            "recipe_found_in_canonical_population": group is not None,
            "recipe_is_anomalous": group["is_anomalous_recipe"] if group else None,
            "all_renders_of_recipe_anomalous": group["all_renders_anomalous"] if group else None,
        })
    members = [entry for entry in results if entry["recipe_found_in_canonical_population"]]
    all_members = bool(members) and all(entry["recipe_is_anomalous"] is True for entry in members)
    return {
        "candidates": results,
        "three_q_mismatches_are_members_of_recipe_level_anomaly_class": all_members,
    }


def c6_selection_impact_for_canonical_anomalies(anomaly_rows: list[dict[str, Any]],
                                                bank_lock: dict[str, Any]) -> dict[str, Any]:
    """TASK I: joins the canonical anomalies against the frozen
    C6_BANK_LOCK_LLM.json 'selected' list (the only per-candidate C6 outcome
    persisted anywhere) and groups the result by recipe. 'accepted'/
    'rejected' counts for NON-selected candidates are not persisted anywhere
    (an already-established limitation) -- reported honestly as None, never
    guessed or recomputed."""
    from collections import defaultdict

    joined = join_with_c6_bank_lock(anomaly_rows, bank_lock)
    selected = sum(1 for row in joined if row["c6_selected"])
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"selected": 0, "not_selected_or_unknown": 0})
    for row in joined:
        bucket = grouped[row.get("recipe_id")]
        bucket["selected" if row["c6_selected"] else "not_selected_or_unknown"] += 1
    return {
        "available": True,
        "canonical_anomaly_c6_selected": selected,
        "canonical_anomaly_c6_not_selected_or_unknown": len(joined) - selected,
        "canonical_anomaly_c6_accepted": None,
        "canonical_anomaly_c6_rejected": None,
        "by_recipe": {recipe_id: dict(counts) for recipe_id, counts in grouped.items()},
        "note": "accepted-vs-rejected is not persisted for non-selected candidates in any frozen "
               "artifact; only SELECTED is directly observable from C6_BANK_LOCK_LLM.json.",
    }


def reassess_root_cause(*, anomaly_determined_by_recipe: str, same_across_routes: bool,
                        same_across_bindings: bool, alternate_value_present_in_recipe: bool,
                        alternate_value_present_in_graph: bool) -> dict[str, Any]:
    """TASK J: root-cause reassessment as a PURE function of the canonical
    evidence this turn's tasks produced -- never a judgment call made after
    seeing which label 'sounds right'."""
    r3_weak = bool(anomaly_determined_by_recipe == "TRUE" and same_across_routes and same_across_bindings)
    if alternate_value_present_in_recipe or alternate_value_present_in_graph:
        primary = "R5_deterministic_recipe_specific_edge_or_fallback_semantic"
        confidence = "HIGH"
    elif r3_weak:
        primary = ("R5_deterministic_recipe_specific_edge_or_fallback_semantic (no exact field match "
                  "found yet; fully deterministic, recipe-constant, route- and binding-independent "
                  "behavior strongly rules R3 down)")
        confidence = "MEDIUM"
    elif anomaly_determined_by_recipe == "TRUE":
        primary = "R1_historical_runtime_semantic_not_captured"
        confidence = "LOW"
    else:
        primary = "UNRESOLVED"
        confidence = "LOW"
    return {
        "primary_anomaly_factor": primary,
        "r3_mutable_runtime_state_ruled_weak": r3_weak,
        "root_cause_confidence": confidence,
    }


def run_canonical_anomaly_investigation(repo: Path) -> dict[str, Any]:
    """TASKS A-J, orchestrated: strictly read-only characterization of the
    CANONICAL population's own sparse anomalies. Operates on
    E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv only; the UNFILTERED
    3058-row tree is touched ONLY for the TASK F cross-binding check, and its
    rows are never folded into any scientific count. Writes
    E6_CANONICAL_TRACE_ANOMALIES.csv, E6_CANONICAL_TRACE_ANOMALY_SUMMARY.json
    and E6_CANONICAL_ANOMALOUS_RECIPE_FIELD_MATCHES.csv, additive-only, never
    touching the canonical-population artifact itself."""
    import csv as csv_module

    loaded = load_canonical_population_csv(repo)
    out_dir = repo / RENDER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    anomalies_csv_path = out_dir / "E6_CANONICAL_TRACE_ANOMALIES.csv"
    summary_path = out_dir / "E6_CANONICAL_TRACE_ANOMALY_SUMMARY.json"
    field_matches_csv_path = out_dir / "E6_CANONICAL_ANOMALOUS_RECIPE_FIELD_MATCHES.csv"

    if not loaded["available"]:
        summary = {"schema_version": "e6-canonical-trace-anomaly-summary-v1", "available": False,
                  "reason": loaded.get("reason"),
                  "note": "the canonical population CSV is not present on this host; no anomaly CSV "
                          "was written, and none was fabricated."}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"available": False, "reason": loaded.get("reason"), "csv_written": False,
               "summary_written": True, "csv_path": None, "summary_path": str(summary_path)}

    rows = loaded["rows"]
    anomalies = filter_anomalies(rows)
    groups = summarize_canonical_recipe_groups(rows, anomalies)
    anomaly_determined = classify_anomaly_determined_by_recipe(groups)
    anomalous_recipe_ids = {group["recipe_id"] for group in groups if group["is_anomalous_recipe"]}

    recipe_field_matches = compare_anomalous_recipe_fields(repo, groups)
    graph_field_matches = compare_anomalous_compiled_graph_fields(repo, groups)
    source_classification = classify_alternate_coverage_source(recipe_field_matches, graph_field_matches)

    unfiltered_audit = audit_historical_trace_population(repo)
    canonical_resolution = resolve_canonical_gpat_binding(repo)
    consistency = cross_route_and_binding_consistency(
        rows, unfiltered_audit.get("rows") or [],
        canonical_gpat_binding=canonical_resolution.get("canonical_gpat_binding"),
        second_gpat_binding=canonical_resolution.get("second_gpat_binding"),
        anomalous_recipe_ids=anomalous_recipe_ids)

    controls = select_normal_control_recipes(rows, anomalous_recipe_ids=anomalous_recipe_ids)
    known_relation = relate_known_q_mismatches_to_recipe_class(groups)

    bank_lock_path = repo / C6_BANK_LOCK_LLM_PATH
    if bank_lock_path.is_file():
        bank_lock = json.loads(bank_lock_path.read_text(encoding="utf-8"))
        c6_impact = c6_selection_impact_for_canonical_anomalies(anomalies, bank_lock)
    else:
        c6_impact = {"available": False}

    root_cause = reassess_root_cause(
        anomaly_determined_by_recipe=anomaly_determined,
        same_across_routes=consistency["same_recipe_same_alternate_coverage_across_routes"],
        same_across_bindings=consistency["same_recipe_same_alternate_coverage_across_gpat_bindings"],
        alternate_value_present_in_recipe=source_classification["alternate_value_present_in_recipe"],
        alternate_value_present_in_graph=source_classification["alternate_value_present_in_compiled_graph"])

    fieldnames = list(anomalies[0].keys()) if anomalies else []
    with anomalies_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in anomalies:
            writer.writerow(row)

    recipe_by_id = {match["recipe_id"]: match for match in recipe_field_matches}
    graph_by_id = {match["recipe_id"]: match for match in graph_field_matches}
    field_match_rows = []
    for recipe_id in sorted(anomalous_recipe_ids):
        recipe_match = recipe_by_id.get(recipe_id, {})
        graph_match = graph_by_id.get(recipe_id, {})
        field_match_rows.append({
            "recipe_id": recipe_id,
            "recipe_geometry_coverage": recipe_match.get("recipe_geometry_coverage"),
            "historical_trace_requested_coverage": recipe_match.get("historical_trace_requested_coverage"),
            "matching_recipe_fields_exact": ";".join(recipe_match.get("matching_recipe_fields_exact") or []),
            "matching_recipe_fields_tolerance": ";".join(recipe_match.get("matching_recipe_fields_tolerance") or []),
            "matching_graph_fields_exact": ";".join(graph_match.get("matching_graph_fields_exact") or []),
            "matching_graph_fields_tolerance": ";".join(graph_match.get("matching_graph_fields_tolerance") or []),
        })
    field_match_fieldnames = list(field_match_rows[0].keys()) if field_match_rows else [
        "recipe_id", "recipe_geometry_coverage", "historical_trace_requested_coverage",
        "matching_recipe_fields_exact", "matching_recipe_fields_tolerance",
        "matching_graph_fields_exact", "matching_graph_fields_tolerance"]
    with field_matches_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=field_match_fieldnames)
        writer.writeheader()
        for row in field_match_rows:
            writer.writerow(row)

    summary = {
        "schema_version": "e6-canonical-trace-anomaly-summary-v1", "available": True,
        "canonical_population_total": len(rows),
        "canonical_anomaly_count": len(anomalies),
        "canonical_gpat_anomaly_count": sum(1 for row in anomalies if row.get("route") == "gpat"),
        "canonical_physics_anomaly_count": sum(1 for row in anomalies if row.get("route") == "physics"),
        "unique_canonical_anomalous_recipes": len(anomalous_recipe_ids),
        "anomalous_recipe_ids": sorted(anomalous_recipe_ids),
        "recipe_groups": groups,
        "anomaly_determined_by_recipe_id": anomaly_determined,
        "recipe_field_matches": recipe_field_matches,
        "graph_field_matches": graph_field_matches,
        "alternate_coverage_source_classification": source_classification,
        "cross_route_and_binding_consistency": consistency,
        "normal_control_recipes": controls,
        "known_q_mismatch_relation": known_relation,
        "c6_selection_impact": c6_impact,
        "root_cause_reassessment": root_cause,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"available": True, "csv_written": True, "summary_written": True,
           "csv_path": str(anomalies_csv_path), "field_matches_csv_path": str(field_matches_csv_path),
           "summary_path": str(summary_path), "summary": summary}


def gpu_canonical_anomaly_investigation_command() -> str:
    """TASKS A-J: prepared, never executed here. Read-only on the GPU host."""
    return "python -m prism_fas.evaluation.c_ext_e6_render --analyze-canonical-trace-anomalies"


def run_render_execution(repo: Path, *,
                         candidate_renderer: Callable[..., dict[str, Any]] | None = None,
                         quality_matcher: Callable[..., dict[str, Any]] | None = None,
                         metrics_provider: Callable[..., dict[str, Any]] | None = None
                         ) -> dict[str, str]:
    """TASK D, steps 1-15. Builds everything in memory and stages every
    candidate BEFORE any promotion; the final bank lock is written LAST, and
    a failure at any step leaves no usable bank lock (`is_usable_bank_lock`
    recomputes rather than trusts).
    """
    # 1-2. validate the frozen locks + the shuffled recipe artifact
    shuffle = verify_shuffle_recipe_source(repo)
    # 3. resolve historical source package/store/routes (never instantiates a route)
    runtime = resolve_gpu_runtime(repo)
    if not runtime["c4_lock_ok"] or not runtime["SOURCE_STORE_RESOLVABLE"] or not runtime["CUDA_AVAILABLE"]:
        raise E6RenderError(
            f"GPU runtime is not fully resolvable (c4_lock_ok={runtime['c4_lock_ok']}, "
            f"SOURCE_STORE_RESOLVABLE={runtime['SOURCE_STORE_RESOLVABLE']}, "
            f"CUDA_AVAILABLE={runtime['CUDA_AVAILABLE']}); refusing to render")
    # TASK I: the quality backend stack must resolve BEFORE a single candidate
    # is rendered -- a 2048-candidate GPU render must never run against a
    # quality pipeline already known to be unavailable.
    if metrics_provider is None and not runtime["QUALITY_BACKENDS_RESOLVABLE"]:
        raise E6RenderError(
            "the quality backend stack is not fully resolvable "
            f"(QUALITY_BACKENDS_RESOLVABLE=False); refusing to render before quality gating can run. "
            "See resolve_quality_backend_assets() for the exact missing dependency.")
    # 4. construct the E6 arm plan from the frozen shuffled recipes
    plan = build_render_plan(repo)
    build_parity_table(repo, plan)  # fails closed on any non-recipe difference
    original_recipes = cc.read_jsonl(repo / "assets/recipe_banks/c3/llm/recipes.jsonl")
    alignment = verify_source_pair_recipe_alignment(
        repo, original_recipes=original_recipes, shuffled_recipes=shuffle["recipes"])
    alignment_lock = write_source_pair_alignment_lock(repo, alignment)

    # 5-7. render (staged; resume-safe via reuse_decision) and validate the count
    staged = render_candidates_to_staging(repo=repo, plan=plan, recipes=shuffle["recipes"],
                                          candidate_renderer=candidate_renderer)
    if len(staged["rows"]) != plan["expected_candidate_count"]:
        raise E6RenderError(
            f"staged {len(staged['rows'])} candidates, expected {plan['expected_candidate_count']}")

    # 8-11. quality score, gate, C6-match, select exactly the matched-bank count
    matcher = quality_matcher or default_quality_matcher
    matcher_kwargs = ({"metrics_provider": metrics_provider, "arm": E6_ARM_NAME}
                      if matcher is default_quality_matcher else {})
    matched = matcher(repo=repo, plan=plan, staged=staged, **matcher_kwargs)
    selected = matched["selected"]
    q_audit = build_q_audit(original_llm_q=matched["original_llm_q"],
                            shuffle_a_q_values=[float(row["q"]) for row in selected])

    # 12. validate final bank with the EXISTING C6MatchedBankReader/schema
    bank_lock = build_matched_bank_lock(plan=plan, selected=selected,
                                        source_pair_alignment_lock=alignment_lock, q_audit=q_audit)
    verify_bank_readable_by_c6_matched_bank_reader(
        candidates_root=repo / CANDIDATES_ROOT, bank_lock=bank_lock, recipes=shuffle["recipes"])

    # 13-15. freeze q summary, promote transactionally, publish the bank lock LAST
    written = {
        "source_pair_alignment_lock": cc.write_json_atomic(
            SOURCE_PAIR_ALIGNMENT_LOCK_PATH, alignment_lock, root=repo),
        "q_audit": cc.write_json_atomic(Q_AUDIT_PATH, q_audit, root=repo),
        "bank_lock": cc.write_json_atomic(BANK_LOCK_PATH, bank_lock, root=repo),
    }
    return written


# --------------------------------------------------------------------------- #
# GPU command preparation (never executed here)
# --------------------------------------------------------------------------- #

def files_to_sync_to_gpu() -> list[str]:
    return [
        "src/prism_fas/evaluation/c_ext_e6_render.py",
        "src/prism_fas/evaluation/c_ext_e6_training_plan.py",
        "src/prism_fas/evaluation/c_ext_llm_shuffle.py",
        "src/prism_fas/evaluation/c_ext_quality_reconstruct.py",
        "reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_TRAINING_PLAN_LOCK.json",
        "reports/c_ext_q1q2_v1/e6_llm_shuffle/LLM_SHUFFLE_A_RECIPES.jsonl",
        "reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_LLM_SHUFFLE_A.json",
        "reports/c_ext_q1q2_v1/e6_llm_shuffle/render/E6_RENDER_PLAN_LOCK.json",
        # Already GPU-host-resident (never re-synced/regenerated); named here so
        # the GPU-side preflight/render/audit commands' preconditions are explicit.
        QUALITY_CALIBRATION_PATH,
    ]


def gpu_preflight_command() -> str:
    return "python -m prism_fas.evaluation.c_ext_e6_render --preflight"


def gpu_render_command() -> str:
    return "python -m prism_fas.evaluation.c_ext_e6_render --execute --authorize-gpu-render"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6 LLM-SHUFFLE-A render adapter (no LLM, no target)")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-gpu-render", action="store_true")
    parser.add_argument("--historical-q-audit", action="store_true",
                        help="TASK G: read-only. Recomputes q for a sample of the frozen "
                             "ORIGINAL_LLM selected candidates and compares it to the persisted "
                             "value. Never renders, never trains, never touches a historical "
                             "artifact. DEFERRED (not FAIL) if the quality backends or the real "
                             "historical candidate bytes are not present on this host.")
    parser.add_argument("--diagnose-historical-candidate", nargs="+", default=None, metavar="CANDIDATE_ID",
                        help="TASK C/F: read-only. Recomputes and prints every raw metric, gate "
                             "decision and q for one or more specific frozen ORIGINAL_LLM candidate "
                             "ids. Never renders, never trains, never touches a historical artifact.")
    parser.add_argument("--diagnose-arm", default="LLM",
                        help="arm to diagnose candidates against (default LLM).")
    parser.add_argument("--audit-shard-vs-loose", nargs="+", default=None, metavar="SAMPLE_ID",
                        help="TASK G: read-only. For one or more source_train sample ids, compares the "
                             "loose file SampleStore reads against the frozen shard TAR member the "
                             "package's own content_identity_sha256 covers. Never extracts to disk, "
                             "never repairs, never overwrites.")
    parser.add_argument("--audit-historical-trace-population", action="store_true",
                        help="TASK A: read-only, population-wide. Recomputes ONLY recipe/graph hashes "
                             "(pure compile_recipe -- no image, no model, no GPU) for every frozen "
                             "GENERATED candidate of --audit-arm and writes "
                             "E6_HISTORICAL_TRACE_POPULATION.csv / _SUMMARY.json under this "
                             "extension's own render/ namespace. Never renders, never trains, never "
                             "touches target, never calls an LLM, never instantiates a quality backend.")
    parser.add_argument("--audit-arm", default="LLM",
                        help="arm for --audit-historical-trace-population (default LLM; only LLM has a "
                             "resolvable historical bank in this module).")
    parser.add_argument("--analyze-historical-trace-anomalies", action="store_true",
                        help="TASK A-L: read-only, population-wide. Re-runs the population audit fresh, "
                             "computes the route-count integrity check, filters and enriches every "
                             "requested_equals_recipe_coverage==False row, groups by recipe/live-sample/"
                             "position-mod-8, finds contiguous position blocks and max-deviation "
                             "candidates, categorizes requested-vs-achieved coverage, and joins the "
                             "frozen C6_BANK_LOCK_LLM.json where resolvable. Writes "
                             "E6_HISTORICAL_TRACE_ANOMALIES.csv / _SUMMARY.json. Never renders, never "
                             "trains, never touches target, never instantiates a quality backend.")
    parser.add_argument("--investigate-gpat-binding", action="store_true",
                        help="TASKS A-H: read-only. Aggregates the population by (route, "
                             "route_binding), groups GPAT candidates by the FROZEN_SCHEDULE_KEY "
                             "(position), classifies the DOUBLE_GPAT_RENDER_PASS, resolves which "
                             "GPAT binding is canonical from frozen C4/C5/C6 locks only, explains "
                             "candidate_id's dependency on route_binding, and classifies the 3 known "
                             "q-mismatch candidates' bindings. Writes E6_GPAT_BINDING_POPULATION.csv/"
                             "_SUMMARY.json, E6_GPAT_SCHEDULE_KEY_PAIRS.json, "
                             "E6_GPAT_BINDING_INVESTIGATION_SUMMARY.json, and (only if a canonical "
                             "binding is proven) E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv/"
                             "_SUMMARY.json. Never renders, never trains, never touches target, never "
                             "deletes or mutates a historical candidate or lock.")
    parser.add_argument("--analyze-canonical-trace-anomalies", action="store_true",
                        help="TASKS A-J: read-only. Characterizes the sparse anomalies within "
                             "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv ONLY (never the unfiltered "
                             "3058-row tree, except read-only for the cross-binding check): per-recipe "
                             "grouping, recursive recipe/compiled-graph scalar-field matching against "
                             "trace.requested_coverage, cross-route and cross-GPAT-binding consistency, "
                             "matched normal controls, the 3 known q-mismatch candidates' recipe-class "
                             "membership, C6 selection impact, and a root-cause reassessment. Writes "
                             "E6_CANONICAL_TRACE_ANOMALIES.csv, E6_CANONICAL_TRACE_ANOMALY_SUMMARY.json, "
                             "E6_CANONICAL_ANOMALOUS_RECIPE_FIELD_MATCHES.csv. Never renders, never "
                             "trains, never touches target, never recomputes q, never mutates the "
                             "canonical-population artifact.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.preflight:
        report = run_preflight(repo)
        write_e6_render_preparation(repo)
        print(json.dumps(report, default=str))
        return 0

    if args.historical_q_audit:
        print(json.dumps(historical_q_reproduction_status(repo), default=str))
        return 0

    if args.diagnose_historical_candidate:
        batch = diagnose_historical_candidates(repo, args.diagnose_historical_candidate, arm=args.diagnose_arm)
        print(json.dumps(batch, indent=2, default=str))
        return 0

    if args.audit_shard_vs_loose:
        results = [shard_vs_loose_byte_audit(repo, sample_id) for sample_id in args.audit_shard_vs_loose]
        print(json.dumps(results, indent=2, default=str))
        return 0

    if args.audit_historical_trace_population:
        status = write_historical_trace_population_artifacts(repo, arm=args.audit_arm)
        report = {key: value for key, value in status.items() if key != "rows"}
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.analyze_historical_trace_anomalies:
        status = write_anomaly_artifacts(repo, arm=args.audit_arm)
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.investigate_gpat_binding:
        status = run_gpat_binding_investigation(repo, arm=args.audit_arm)
        report = {key: value for key, value in status.items() if key != "rows"}
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.analyze_canonical_trace_anomalies:
        status = run_canonical_anomaly_investigation(repo)
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.execute:
        if not args.authorize_gpu_render:
            print("--execute requires --authorize-gpu-render as an explicit second flag.")
            return 2
        try:
            written = run_render_execution(repo)
        except E6RenderError as error:
            print(f"E6 render execution refused: {error}")
            return 1
        print(json.dumps(written))
        return 0

    print("Pass --preflight to verify/lock the render plan (no rendering), or --execute "
         "--authorize-gpu-render on the GPU host.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
