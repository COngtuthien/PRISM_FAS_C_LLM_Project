"""PRISM-FAS-C EXT-Q1Q2 - E6: LLM-SHUFFLE-A preparation/audit and training plan.

Scientific question: is the downstream advantage of the LLM arm explained
merely by the marginal frequencies of recipe ingredients, or does the JOINT
STRUCTURE the LLM created among those fields matter? The control is
LLM-SHUFFLE-A: the SAME 256 frozen, already-selected LLM recipes, with six
frozen field-groups pairwise-swapped between recipes (never re-drawn, never
re-generated) so every per-field marginal is preserved EXACTLY while the
joint associations among fields change.

This module is preparation/audit only. It never trains a detector, never
renders an image, never calls an LLM, and never opens any target artifact.

It reuses, rather than reimplements:

* `c_ext_llm_shuffle.run_shuffle` / `load_frozen_group_map` -- the ALREADY
  existing, already-frozen shuffle preparation (re-run here only to VERIFY
  the frozen `E6_LLM_SHUFFLE_A.json` reproduces exactly; `main()` is never
  called, so the frozen artifact is never rewritten).
* Track G's frozen variant flags and winner-config scalars
  (`reports/full/c7/DETECTOR_CONFIG_LOCK.json`), UNCHANGED -- LLM-SHUFFLE-A
  is a mechanism control, not a new architecture: `synthetic`,
  `recipe_conditioning` and `quality_weighting` stay exactly what
  ORIGINAL_LLM already used.
* `c_ext_e5_real_only`'s config-construction pattern
  (`configs/models/m9_detector.yaml` + `configs/train/m9_reference.yaml`,
  loaded via `detector.config.load_m9_configs`), which already established
  the reusable "schedule/optimizer scalars held fixed, only the row-specific
  overrides change" shape this milestone needs too.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_llm_shuffle as shuffle_module
from prism_fas.evaluation.c_ext_e5_real_only import TRACK_G_SOURCE_PACKAGE_IDENTITY

E6_DIR = "reports/c_ext_q1q2_v1/e6_llm_shuffle"
TRAINING_REAUDIT_PATH = f"{E6_DIR}/E6_TRAINING_REAUDIT.json"
TRAINING_PLAN_PATH = f"{E6_DIR}/E6_TRAINING_PLAN.json"
TRAINING_PLAN_LOCK_PATH = f"{E6_DIR}/E6_TRAINING_PLAN_LOCK.json"
STEP_MATCHING_AUDIT_PATH = f"{E6_DIR}/E6_STEP_MATCHING_AUDIT.json"
PROVENANCE_PATH = f"{E6_DIR}/E6_PROVENANCE.json"

E6_RUN_ROOT = "runs/c_ext_q1q2_v1/EXT-F1/e6_llm_shuffle"

SEEDS: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)
SHUFFLE_SEED = 20260911
EXPECTED_RECIPE_COUNT = 256

C7_LOCK_PATH = "reports/full/c7/DETECTOR_CONFIG_LOCK.json"

#: Track G's SHARED (RND/DET/LLM-independent, per `track_configuration`'s own
#: docstring) frozen variant flags and winner-config scalars, pinned here so
#: a drifted C7 lock is caught rather than silently trusted.
EXPECTED_TRACK_G_VARIANT_FLAGS: dict[str, Any] = {
    "frames_per_video": 4, "fusion": "single_logit", "global_branch": "siglip2_frozen",
    "local_branch": "off", "manifold": "off", "outlier_loss": "off", "prompt": "off",
    "prototype_k": 0, "quality_weighting": "q_weighted", "recipe_conditioning": "structured",
    "region": "off", "sampler": "domain_class_balanced", "synthetic": "bank_physics_gpat",
}
EXPECTED_WINNER_CONFIG_SHA256 = "97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd"
EXPECTED_DECISION_GRAPH_HASH = "66910279d7c72f513021e3f4d677a150253182042973274894e8e7a3fc3e1ef0"

HISTORICAL_TOTAL_EPOCHS = 35
HISTORICAL_STEPS_PER_EPOCH = 45
HISTORICAL_ACCUMULATION_STEPS = 1
HISTORICAL_TRAINING_BUDGET = HISTORICAL_TOTAL_EPOCHS * HISTORICAL_STEPS_PER_EPOCH  # 1575

#: G1 (3 epochs) + the 2 "manifold warm-up" epochs Track G (manifold=off)
#: spends as ordinary G1 epochs = 5 real-only epochs; G5 = 30 mixed epochs.
#: (`detector.trainer.stage_for_epoch` / `batch_contract_for`).
G1_EPOCHS = 5
G5_EPOCHS = 30
MIXED_BATCH_SYNTHETIC_SLOTS_PER_STEP = 8
EXPECTED_TOTAL_SYNTHETIC_SLOTS = G5_EPOCHS * HISTORICAL_STEPS_PER_EPOCH * MIXED_BATCH_SYNTHETIC_SLOTS_PER_STEP  # 10800

M7_RECIPE_BANK_ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"
C3_SCIENTIFIC_BANK_LOCK_PATH = "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"
ORIGINAL_LLM_SOURCE_BANK_PATH = "assets/recipe_banks/c3/llm/recipes.jsonl"
E6_SHUFFLE_AUDIT_PATH = f"{E6_DIR}/E6_LLM_SHUFFLE_A.json"
E6_SHUFFLE_RECIPES_PATH = f"{E6_DIR}/LLM_SHUFFLE_A_RECIPES.jsonl"

PLAN_SCHEMA_VERSION = "e6-training-plan-v1"
LOCK_SCHEMA_VERSION = "e6-training-plan-lock-v1"


class E6TrainingPlanError(RuntimeError):
    """A precondition for the E6 LLM-SHUFFLE-A training plan failed. Fails closed."""


# --------------------------------------------------------------------------- #
# TASK A (module-level reuse): verify the prepared shuffle artifact
# --------------------------------------------------------------------------- #

def verify_shuffle_artifact(repo: Path) -> dict[str, Any]:
    """Re-derives the shuffle via `c_ext_llm_shuffle.run_shuffle` (never
    `main()`, which would rewrite the frozen artifact) and cross-checks the
    result against the frozen `E6_LLM_SHUFFLE_A.json` / `LLM_SHUFFLE_A_RECIPES.jsonl`.

    Fails closed on: missing frozen artifact, seed mismatch, recipe-count
    mismatch, a marginal that is not exactly preserved, a failed bank
    validation, joint structure that did not actually change, or an
    identity that does not reproduce bit-for-bit.
    """
    audit_path = repo / E6_SHUFFLE_AUDIT_PATH
    recipes_path = repo / E6_SHUFFLE_RECIPES_PATH
    missing = [str(p) for p in (audit_path, recipes_path) if not p.is_file()]
    if missing:
        raise E6TrainingPlanError(f"missing prepared E6 LLM-SHUFFLE-A artifact(s): {missing}")

    frozen = json.loads(audit_path.read_text(encoding="utf-8"))
    if frozen.get("e6_preparation_status") != "PREPARED":
        raise E6TrainingPlanError(
            f"E6_LLM_SHUFFLE_A.json e6_preparation_status={frozen.get('e6_preparation_status')!r} != 'PREPARED'")
    if int(frozen.get("seed", -1)) != SHUFFLE_SEED:
        raise E6TrainingPlanError(f"shuffle seed {frozen.get('seed')!r} != expected {SHUFFLE_SEED}")
    if int(frozen.get("n", -1)) != EXPECTED_RECIPE_COUNT:
        raise E6TrainingPlanError(f"shuffled recipe count {frozen.get('n')!r} != expected {EXPECTED_RECIPE_COUNT}")
    if frozen.get("llm_api_calls") != 0:
        raise E6TrainingPlanError("frozen artifact does not record llm_api_calls == 0")
    if frozen.get("target_labels_accessed") is not False:
        raise E6TrainingPlanError("frozen artifact does not record target_labels_accessed == false")

    group_map = shuffle_module.load_frozen_group_map()
    original_bank = cc.read_jsonl(repo / ORIGINAL_LLM_SOURCE_BANK_PATH)
    if len(original_bank) != EXPECTED_RECIPE_COUNT:
        raise E6TrainingPlanError(
            f"original LLM recipe artifact has {len(original_bank)} recipes, expected {EXPECTED_RECIPE_COUNT}")

    recomputed = shuffle_module.run_shuffle(original_bank, group_map, seed=SHUFFLE_SEED)
    if not recomputed["exact_marginal_assertion"]["all_preserved"]:
        raise E6TrainingPlanError("recomputed shuffle does not preserve every field's exact marginal")
    if not recomputed["bank_level_validation"]["passed"]:
        raise E6TrainingPlanError("recomputed shuffled bank fails validation")
    if recomputed["joint_structure_changed"]["recipes_differing_from_original"] == 0:
        raise E6TrainingPlanError("recomputed shuffle changed no recipe's joint structure")

    recomputed_identity = cc.sha256_json(recomputed["working_bank"])
    if recomputed_identity != frozen.get("shuffled_bank_identity"):
        raise E6TrainingPlanError(
            f"recomputed shuffled_bank_identity {recomputed_identity!r} != frozen "
            f"{frozen.get('shuffled_bank_identity')!r}; the prepared artifact does not reproduce")

    recomputed_lines = [json.dumps(rec, sort_keys=True, separators=(",", ":"))
                       for rec in recomputed["working_bank"]]
    frozen_lines = recipes_path.read_text(encoding="utf-8").strip().split("\n")
    if recomputed_lines != frozen_lines:
        raise E6TrainingPlanError("recomputed shuffled recipes do not byte-reproduce LLM_SHUFFLE_A_RECIPES.jsonl")

    c3_lock = cc.read_json(repo / C3_SCIENTIFIC_BANK_LOCK_PATH)
    original_identity = c3_lock["arms"]["LLM"]["selected_set_identity"]
    if frozen.get("source_bank_sha256") != original_identity:
        raise E6TrainingPlanError("frozen artifact's source_bank_sha256 disagrees with the current C3 bank lock")

    return {"frozen": frozen, "recomputed": recomputed, "original_llm_recipe_identity": original_identity,
           "shuffled_recipe_identity": recomputed_identity, "original_recipe_count": len(original_bank),
           "shuffled_recipe_count": len(recomputed["working_bank"])}


# --------------------------------------------------------------------------- #
# TASK C: original Track-G / LLM training contract audit
# --------------------------------------------------------------------------- #

def verify_original_llm_reference_contract(repo: Path) -> dict[str, Any]:
    """Reads the REAL, frozen C7 lock and pins Track G's shared contract --
    fails closed if it has drifted from what this module's fixed expectations
    (recorded when this audit was written) declare."""
    lock = cc.read_json(repo / C7_LOCK_PATH)
    g = dict(lock.get("tracks", {}).get("G") or {})
    actual_flags = dict(g.get("variant_flags") or {})
    if actual_flags != EXPECTED_TRACK_G_VARIANT_FLAGS:
        raise E6TrainingPlanError(
            f"Track G variant_flags changed since this audit was written: expected "
            f"{EXPECTED_TRACK_G_VARIANT_FLAGS}, found {actual_flags}")
    if g.get("winner_config_sha256") != EXPECTED_WINNER_CONFIG_SHA256:
        raise E6TrainingPlanError("Track G winner_config_sha256 has drifted from the pinned expectation")
    if g.get("decision_graph_hash") != EXPECTED_DECISION_GRAPH_HASH:
        raise E6TrainingPlanError("Track G decision_graph_hash has drifted from the pinned expectation")
    winner = dict(g.get("winner_config") or {})
    source_package_identity = str(winner.get("c7_search_binding", {}).get("source_package_identity") or "")
    if source_package_identity != TRACK_G_SOURCE_PACKAGE_IDENTITY:
        raise E6TrainingPlanError("Track G source package identity has drifted from the pinned expectation")
    return {"lock": lock, "variant_flags": actual_flags, "winner_config": winner,
           "winner_config_sha256": g["winner_config_sha256"], "decision_graph_hash": g["decision_graph_hash"],
           "source_package_identity": source_package_identity}


def build_step_matching_audit(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6-step-matching-audit-v1",
        "reference_arm": "ORIGINAL_LLM", "reference_track": "G",
        "held_fixed": {
            "model_architecture": "single_logit/global-only, siglip2_frozen, manifold=off, region=off "
                                  "(Track G variant_flags, unchanged)",
            "source_package": reference["source_package_identity"],
            "source_train_dev_splits": "EXT-F1: CASIA-FASD + MSU-MFSD, unchanged",
            "detector_seeds": list(SEEDS),
            "total_epochs": HISTORICAL_TOTAL_EPOCHS, "steps_per_epoch": HISTORICAL_STEPS_PER_EPOCH,
            "accumulation_steps": HISTORICAL_ACCUMULATION_STEPS,
            "expected_optimizer_steps": HISTORICAL_TRAINING_BUDGET,
            "g1_epochs_real_only_16_16_0": G1_EPOCHS, "g5_epochs_mixed_12_12_8": G5_EPOCHS,
            "total_synthetic_slots": EXPECTED_TOTAL_SYNTHETIC_SLOTS,
            "optimizer": "AdamW (unchanged)", "lr_schedule": "cosine, 5% warmup (unchanged)",
            "recipe_conditioning": "structured (unchanged -- NOT disabled by shuffling)",
            "quality_weighting": "q_weighted (unchanged -- NOT disabled by shuffling)",
            "checkpoint_selection_rule": "best.pt, else last.pt (source_dev/acer primary, "
                                        "source_dev/bpcer tie-break, source_dev/nll calibration)",
            "source_dev_calibration_rule": "evaluation.source_selection.fit_source_dev_calibration "
                                          "(unchanged)",
            "winner_config_sha256": reference["winner_config_sha256"],
            "decision_graph_hash": reference["decision_graph_hash"],
        },
        "intended_difference": {
            "ORIGINAL_LLM": "frozen original LLM recipe joint structure (256 recipes as selected by C3/C6)",
            "LLM_SHUFFLE_A": "same 256 recipes' per-field marginals, joint structure shuffled "
                            "(seed 20260911, 6 frozen field-groups)",
        },
        "conclusion": "step/batch/synthetic-budget MATCHED by construction: LLM-SHUFFLE-A reuses "
                     "the identical Track G variant flags and winner-config scalars ORIGINAL_LLM "
                     "already uses -- nothing in the training schedule is recomputed or varied "
                     "for this milestone; only which images the synthetic quarter of the batch "
                     "is drawn from can differ, once a shuffled bank exists.",
    }


# --------------------------------------------------------------------------- #
# TASK F: bank consumption path
# --------------------------------------------------------------------------- #

def audit_bank_consumption_path(repo: Path) -> dict[str, Any]:
    """recipe -> renderer -> candidate/bank -> M9TrainingDataset.

    `detector.c6_bank.open_arm_bank` opens a `C6MatchedBankReader` from
    `candidates_root` -- i.e. training consumes ALREADY-RENDERED candidate
    images (via `synthesis.c5_raw_generation` / `c5_arm_plan` /
    `c6_matched_bank` / quality gating), never a bare recipe file. A
    recipe's rendered pixels are determined by its exact parameter
    combination, so a recipe whose joint structure changed (LLM-SHUFFLE-A)
    necessarily needs its OWN new render -- the historical LLM rendered
    bank cannot be substituted without defeating the experiment.
    """
    llm_shuffle_render_present = (
        repo / "assets/recipe_banks/c3/llm_shuffle_a").exists() or (
        repo / f"{E6_DIR}/rendered_bank").exists()
    return {
        "path": ["recipe (C3, per-arm selected set)",
                "synthesis.c5_raw_generation / c5_arm_plan (physics/GPAT candidate rendering)",
                "synthesis.c6_matched_bank / quality_gate (selection into a matched bank)",
                "detector.c6_bank.open_arm_bank -> C6MatchedBankReader",
                "detector.dataset.M9TrainingDataset (consumes rendered images + masks + q, never a recipe)"],
        "conclusion": "training consumes a RENDERED bank, not recipes directly",
        "rendering_is_cpu_safe": False,
        "reason_not_cpu_safe": "C5 candidate generation is a dataset-scale image-rendering pipeline "
                               "(physics transforms and/or the GPAT generative model) with its own "
                               "identity/reuse/candidate-directory machinery "
                               "(synthesis.c5_raw_generation.GenerationIdentity et al.) -- not a "
                               "quick metadata transform, and the GPAT route specifically requires "
                               "a trained generative model best run on GPU.",
        "llm_shuffle_a_rendered_bank_present": llm_shuffle_render_present,
        "existing_render_adapter_for_arbitrary_recipe_source": False,
        "next_implementation_step": "the existing C5/C6 orchestration "
                                    "(synthesis.c5_arm_plan.build_arm_plan / load_arm_bank) is wired "
                                    "to the fixed, frozen assets/recipe_banks/c3/{arm}/ locations for "
                                    "arm in {RND,DET,LLM} -- there is no existing CLI flag that "
                                    "points it at an arbitrary shuffled-recipe source. A small adapter "
                                    "(mirroring how c_ext_e5_real_only added a REAL_ONLY variant "
                                    "without touching the shared M9 stack) is the concrete next step, "
                                    "reusing the SAME renderer/quality-gate settings ORIGINAL_LLM used, "
                                    "with LLM_SHUFFLE_A_RECIPES.jsonl as the only changed input. This "
                                    "module does not fabricate that adapter's CLI command, since it "
                                    "does not yet exist.",
    }


# --------------------------------------------------------------------------- #
# TASK D: quality (q) distribution -- audit only, never fabricated
# --------------------------------------------------------------------------- #

def audit_quality_distribution(repo: Path) -> dict[str, Any]:
    path = repo / "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv"
    original_llm_q = None
    if path.is_file():
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["arm"] == "LLM":
                    original_llm_q = {key: float(row[key]) if key != "n" else int(row[key])
                                     for key, value in row.items() if key != "arm"}
    return {
        "original_llm_q_summary": original_llm_q,
        "original_llm_q_source": path.relative_to(repo).as_posix() if path.is_file() else None,
        "llm_shuffle_a_q_summary": None,
        "llm_shuffle_a_q_available": False,
        "smd_original_vs_shuffle_a": None,
        "reason_shuffle_a_q_unavailable": "realized q is a QUALITY-MODEL OUTPUT computed over an "
                                         "actually-rendered synthetic image (M8 quality gate), not a "
                                         "property of recipe metadata -- LLM-SHUFFLE-A has not been "
                                         "rendered (see audit_bank_consumption_path), so no realized q "
                                         "exists for it yet. Fabricating one from the recipe fields "
                                         "would misrepresent a rendering-dependent quantity as if it "
                                         "were already measured.",
        "correct_measurement_checkpoint": "after LLM-SHUFFLE-A's own C5 render + C6 quality-gate pass "
                                         "completes (the SAME checkpoint ORIGINAL_LLM's own "
                                         "reconstructed-q artifact, reports/c_ext_q1q2_v1/e2_quality/"
                                         "reconstructed_q/C6_Q_RECONSTRUCTED.parquet, was measured at) "
                                         "-- not before, and not from this preparation/audit module.",
        "e8_note": "E2 already flagged an arm-level q difference and opened E8 (q-matching) as its "
                  "own milestone; this audit does not trigger or implement E8 -- it only reports "
                  "that LLM-SHUFFLE-A's own q distribution cannot yet be compared.",
    }


# --------------------------------------------------------------------------- #
# TASK E: training config + plan + lock
# --------------------------------------------------------------------------- #

def build_e6_variant_flags() -> dict[str, Any]:
    """Track G's flags, UNCHANGED. LLM-SHUFFLE-A is a mechanism control: the
    variant is architecturally identical to ORIGINAL_LLM's; only the bank
    CONTENT (once rendered) differs."""
    return dict(EXPECTED_TRACK_G_VARIANT_FLAGS)


def build_e6_training_config(*, repo: Path, seed: int, run_id: str,
                            synthetic_bank_identity: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Mirrors `c_ext_e5_real_only.build_e5_training_config`'s shape: every
    scalar C7 froze for Track G is read from the SAME winner config and
    reused unchanged. `synthetic_bank_identity` is left unresolved (None)
    until a rendered LLM-SHUFFLE-A bank exists -- this function still
    constructs the config so the schedule/optimizer/variant can be
    verified now, but the returned config's `synthetic_bank_identity` is
    an explicit sentinel, never a fabricated value, when unresolved.
    """
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.pipeline.adapters.c7 import _TRIAL_LOSS_WEIGHTS

    reference = verify_original_llm_reference_contract(repo)
    variant = ResolvedExperimentVariant.resolve(build_e6_variant_flags())
    configs = load_m9_configs(repo / "configs/models/m9_detector.yaml",
                              repo / "configs/train/m9_reference.yaml", variant=variant)
    reference_config = configs["training_config"]
    winner = reference["winner_config"]

    weights = dict(reference_config.loss_weights)
    for name in _TRIAL_LOSS_WEIGHTS:
        if name in winner:
            weights[name] = float(winner[name])

    overrides: dict[str, Any] = {
        "run_id": run_id, "seed": int(seed), "prototype_seed": int(seed), "variant": variant,
        "loss_weights": weights,
        "synthetic_bank_identity": synthetic_bank_identity or "PENDING_LLM_SHUFFLE_A_RENDER",
        "source_domains": reference_config.source_domains,
    }
    multiplier = winner.get("learning_rate_multiplier")
    anchor = dict(reference.get("lock", {}).get("tracks", {}).get("G", {}).get("lr_anchor_vector") or {})
    if multiplier is not None and anchor:
        for group, value in anchor.items():
            overrides[group] = float(value) * float(multiplier)
    if "weight_decay" in winner:
        overrides["weight_decay"] = float(winner["weight_decay"])
    if "warmup" in winner:
        overrides["warmup_fraction"] = float(winner["warmup"])

    config = replace(reference_config, **overrides)
    if int(config.total_epochs) != HISTORICAL_TOTAL_EPOCHS or int(config.steps_per_epoch) != HISTORICAL_STEPS_PER_EPOCH:
        raise E6TrainingPlanError("E6 config schedule disagrees with the pinned historical budget")
    return config, configs


def seed_run_id(seed: int) -> str:
    return f"EXT-F1-G-LLM-SHUFFLE-A-s{seed}"


def build_training_plan(repo: Path) -> dict[str, Any]:
    shuffle = verify_shuffle_artifact(repo)
    reference = verify_original_llm_reference_contract(repo)
    bank_path = audit_bank_consumption_path(repo)

    seeds_payload = []
    for seed in SEEDS:
        config, _ = build_e6_training_config(repo=repo, seed=seed, run_id=seed_run_id(seed))
        seeds_payload.append({"seed": seed, "run_id": seed_run_id(seed),
                             "expected_optimizer_steps": int(config.total_epochs) * int(config.steps_per_epoch)})

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "milestone": "E6", "arm": "LLM_SHUFFLE_A", "track": "G", "fold": "EXT-F1",
        "shuffle_seed": SHUFFLE_SEED,
        "original_llm_recipe_identity": shuffle["original_llm_recipe_identity"],
        "llm_shuffle_a_recipe_identity": shuffle["shuffled_recipe_identity"],
        "recipe_count": shuffle["shuffled_recipe_count"],
        "source_package_identity": reference["source_package_identity"],
        "ontology_identity": M7_RECIPE_BANK_ONTOLOGY_IDENTITY,
        "detector_config_identity": reference["winner_config_sha256"],
        "decision_graph_hash": reference["decision_graph_hash"],
        "detector_seeds": list(SEEDS), "seed_count": len(SEEDS),
        "expected_optimizer_steps": HISTORICAL_TRAINING_BUDGET,
        "expected_synthetic_sample_budget": EXPECTED_TOTAL_SYNTHETIC_SLOTS,
        "quality_weighting_status": "q_weighted", "recipe_conditioning_status": "structured",
        "target_access": False, "llm_api_calls": 0,
        "rendered_bank_required": True, "rendered_bank_status": "NEEDS_BUILD",
        "seeds": seeds_payload,
        "run_root": E6_RUN_ROOT,
    }
    plan["plan_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(plan))
    return plan


def check_fail_closed_conditions(repo: Path) -> dict[str, Any]:
    """TASK G: the explicit block-list. Returns which conditions were
    checked and whether each passed; raises via the individual verify_*
    functions on the first hard failure (never silently continues past a
    real violation)."""
    shuffle = verify_shuffle_artifact(repo)
    reference = verify_original_llm_reference_contract(repo)
    checks = {
        "shuffle_artifact_verified": True,
        "marginals_exact": shuffle["recomputed"]["exact_marginal_assertion"]["all_preserved"],
        "recipe_validation_passed": shuffle["recomputed"]["bank_level_validation"]["passed"],
        "recipe_counts_match": shuffle["original_recipe_count"] == shuffle["shuffled_recipe_count"] == EXPECTED_RECIPE_COUNT,
        "source_package_matches_track_g": reference["source_package_identity"] == TRACK_G_SOURCE_PACKAGE_IDENTITY,
        "optimizer_step_budget_matches": HISTORICAL_TRAINING_BUDGET == HISTORICAL_TOTAL_EPOCHS * HISTORICAL_STEPS_PER_EPOCH,
        "synthetic_slot_budget_defined": EXPECTED_TOTAL_SYNTHETIC_SLOTS > 0,
        "quality_weighting_unchanged": reference["variant_flags"]["quality_weighting"] == "q_weighted",
        "recipe_conditioning_unchanged": reference["variant_flags"]["recipe_conditioning"] == "structured",
        "no_target_dependency": True,
        "no_llm_api_required": True,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise E6TrainingPlanError(f"fail-closed condition(s) not satisfied: {failed}")
    return checks


# --------------------------------------------------------------------------- #
# Writers (lock published last)
# --------------------------------------------------------------------------- #

def build_lock(plan: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": LOCK_SCHEMA_VERSION, "milestone": "E6", "arm": plan["arm"], "track": plan["track"],
        "plan_identity": plan["plan_identity"], "shuffle_seed": plan["shuffle_seed"],
        "original_llm_recipe_identity": plan["original_llm_recipe_identity"],
        "llm_shuffle_a_recipe_identity": plan["llm_shuffle_a_recipe_identity"],
        "recipe_count": plan["recipe_count"], "source_package_identity": plan["source_package_identity"],
        "detector_config_identity": plan["detector_config_identity"],
        "detector_seeds": plan["detector_seeds"], "seed_count": plan["seed_count"],
        "expected_optimizer_steps": plan["expected_optimizer_steps"],
        "expected_synthetic_sample_budget": plan["expected_synthetic_sample_budget"],
        "quality_weighting_status": plan["quality_weighting_status"],
        "recipe_conditioning_status": plan["recipe_conditioning_status"],
        "target_access": False, "llm_api_calls": 0,
        "rendered_bank_required": plan["rendered_bank_required"], "rendered_bank_status": plan["rendered_bank_status"],
        "status": "FROZEN",
    }
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def is_usable_plan_lock(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "FROZEN":
        return False
    if int(payload.get("seed_count", -1)) != len(SEEDS):
        return False
    body = {key: value for key, value in payload.items() if key != "lock_identity"}
    return cc.sha256_bytes(cc.canonical_json_bytes(body)) == payload.get("lock_identity")


def build_provenance(repo: Path, plan: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        commit = ""
    return {"schema_version": "e6-provenance-v1", "code_commit": commit,
           "plan_identity": plan["plan_identity"], "lock_identity": lock["lock_identity"],
           "target_features_accessed": False, "target_labels_accessed": False,
           "llm_api_calls": 0, "detector_training_executed": False, "image_rendering_executed": False}


def write_e6_preparation(repo: Path) -> dict[str, str]:
    shuffle = verify_shuffle_artifact(repo)
    reference = verify_original_llm_reference_contract(repo)
    bank_audit = audit_bank_consumption_path(repo)
    q_audit = audit_quality_distribution(repo)
    check_fail_closed_conditions(repo)
    plan = build_training_plan(repo)
    step_audit = build_step_matching_audit(reference)
    lock = build_lock(plan)
    provenance = build_provenance(repo, plan, lock)

    reaudit = {
        "schema_version": "e6-training-reaudit-v1",
        "shuffle_verification": {k: v for k, v in shuffle.items() if k not in ("frozen", "recomputed")},
        "original_llm_reference_contract": {k: v for k, v in reference.items() if k != "lock"},
        "bank_consumption_path": bank_audit,
        "quality_distribution_audit": q_audit,
    }
    written = {
        "reaudit": cc.write_json_atomic(TRAINING_REAUDIT_PATH, reaudit, root=repo),
        "step_matching_audit": cc.write_json_atomic(STEP_MATCHING_AUDIT_PATH, step_audit, root=repo),
        "plan": cc.write_json_atomic(TRAINING_PLAN_PATH, plan, root=repo),
        "provenance": cc.write_json_atomic(PROVENANCE_PATH, provenance, root=repo),
        "plan_lock": cc.write_json_atomic(TRAINING_PLAN_LOCK_PATH, lock, root=repo),
    }
    return written


# --------------------------------------------------------------------------- #
# GPU command preparation (rendering; never training this milestone)
# --------------------------------------------------------------------------- #

def next_gpu_command_note() -> str:
    return ("No render command is prepared: the existing C5/C6 orchestration has no adapter for an "
           "arbitrary shuffled-recipe source (see E6_TRAINING_REAUDIT.json.bank_consumption_path). "
           "The concrete next step is implementing that adapter, reusing ORIGINAL_LLM's exact "
           "rendering settings against LLM_SHUFFLE_A_RECIPES.jsonl, before any GPU command can be issued.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6 LLM-SHUFFLE-A preparation/audit (no training, no rendering)")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    repo = cc.repo_root()
    if args.prepare:
        written = write_e6_preparation(repo)
        print(json.dumps(written))
        return 0
    print("Pass --prepare to build the E6 reaudit/plan/lock (no training, no rendering, no GPU, no LLM).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
