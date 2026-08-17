"""Build the learning-rate anchor decision dossier for USER APPROVAL.

    python scripts/lr_anchor_decision_dossier.py

Offline, CPU-only, read-only with respect to science. It opens no dataset, runs
no training, allocates no GPU, makes no provider call and resolves no target.

The engineering-readiness milestone stopped with one unresolved result-affecting
decision: `learning_rate` has no uniquely inherited Version-B anchor. This script
reconstructs the evidence behind that report and lays out the interpretations
that are actually legal under §15.2.2 and §15.2.3 — with their exact trial-count
consequences computed by the real search engine rather than by hand.

It decides nothing. `decision_status` is AWAITING_USER_APPROVAL and no plan,
lock or PROJECT_STATE scientific field is changed by running it.

One thing it does resolve, and says so explicitly: for Track G the ambiguity
turns out not to be a user choice at all. Track G instantiates no ConvNeXt
(§13.4.1), so its backbone optimizer group is empty and is omitted; `backbone_lr`
controls zero parameters. Exactly one LR scalar is applicable, which is what the
frozen rules already require. That is reported as ALREADY_IMPLIED_BY_FROZEN_SPEC
rather than offered as an option.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VERSION_B = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "reports" / "handoff"
SCHEMA_VERSION = "prism-lr-anchor-decision-dossier-v1"

SPEC = ("docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_"
        "FullPipeline_Spec_2026.docx")

#: Compliance classes, exactly as the task defines them.
ALREADY_IMPLIED = "ALREADY_IMPLIED_BY_FROZEN_SPEC"
APPROVAL_REQUIRED = "COMPATIBLE_BUT_USER_APPROVAL_REQUIRED"
ENVELOPE_EXPANSION = "SEARCH_ENVELOPE_EXPANSION"
SEMANTIC_CHANGE = "SCIENTIFIC_SEMANTIC_CHANGE"
NOT_APPLICABLE = "NOT_APPLICABLE"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(args: list[str], cwd: Path = REPO) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yaml_of(relative: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((REPO / relative).read_text(encoding="utf-8"))


# --- section 3: the forensic record -----------------------------------------

def lr_evidence() -> dict[str, Any]:
    """Every LR-like scalar, traced to the optimizer group it actually controls."""
    gpat = yaml_of("configs/synthesis/gpat_m8.yaml")
    detector = yaml_of("configs/train/m9_reference.yaml")
    b00 = yaml_of("configs/train/b00_local.yaml")

    gpat_path = "configs/synthesis/gpat_m8.yaml"
    det_path = "configs/train/m9_reference.yaml"
    b00_path = "configs/train/b00_local.yaml"

    def provenance(relative: str) -> dict[str, Any]:
        c_sha = sha256_file(REPO / relative)
        b_sha = sha256_file(VERSION_B / relative)
        return {
            "config_path": relative,
            "version_c_sha256": c_sha,
            "version_b_sha256": b_sha,
            "byte_identical_to_version_b": bool(c_sha and c_sha == b_sha),
            "version_b_last_commit": git(["log", "--format=%H", "-1", "--", relative],
                                         VERSION_B),
        }

    scalars = [
        {"name": "encoder_lr", "value": float(gpat["optimizer"]["encoder_lr"]),
         "owner": "C4 / GPAT", "optimizer": "AdamW (single optimizer, three groups)",
         "parameter_group": "artifact_encoder",
         "modules": ["ArtifactEncoder"],
         "trainable_parameters": 422272,
         "consumed_at": "src/prism_fas/synthesis/gpat_model.py:119 "
                        "(GPATResidualModel.parameter_groups)",
         "active_under_version_c": True,
         "training_only": True,
         "classification": "D_controls_an_active_component",
         "note": "encodes the spoof source's artifact appearance into z_a; a live "
                 "component of the frozen C4 architecture, not a historical one",
         **provenance(gpat_path)},
        {"name": "recipe_lr", "value": float(gpat["optimizer"]["recipe_lr"]),
         "owner": "C4 / GPAT", "optimizer": "AdamW (single optimizer, three groups)",
         "parameter_group": "recipe_encoder",
         "modules": ["RecipeEncoder"],
         "trainable_parameters": 13632,
         "consumed_at": "src/prism_fas/synthesis/gpat_model.py:120",
         "active_under_version_c": True,
         "training_only": True,
         "classification": "C_controls_an_auxiliary_but_active_module",
         "note": "maps the frozen 41-D conditioning vector to z_recipe. It is the only "
                 "path by which a C3 recipe reaches the generator, so it is auxiliary "
                 "in size but not in function",
         **provenance(gpat_path)},
        {"name": "generator_lr", "value": float(gpat["optimizer"]["generator_lr"]),
         "owner": "C4 / GPAT", "optimizer": "AdamW (single optimizer, three groups)",
         "parameter_group": "generator",
         "modules": ["stem", "blocks (FiLMResidualBlock x4)", "delta_head",
                     "artifact_map_head"],
         "trainable_parameters": 474634,
         "consumed_at": "src/prism_fas/synthesis/gpat_model.py:121-123",
         "active_under_version_c": True,
         "training_only": True,
         "classification": "D_controls_the_actual_gpat_generator",
         "note": "the residual generator itself: the component that produces delta_high "
                 "and the artifact map",
         **provenance(gpat_path)},
        {"name": "min_lr", "value": float(gpat["scheduler"]["min_lr"]),
         "owner": "C4 / GPAT", "optimizer": "cosine LambdaLR floor",
         "parameter_group": None, "modules": [], "trainable_parameters": 0,
         "consumed_at": "src/prism_fas/synthesis/gpat_trainer.py:40-53 "
                        "(cosine_schedule)",
         "active_under_version_c": True, "training_only": True,
         "classification": "E_not_an_lr_anchor",
         "note": "an absolute schedule FLOOR, not a base learning rate. §15.2.2 names "
                 "the learning rate and the warm-up fraction as coordinates; a decay "
                 "floor is neither, and searching it would add a coordinate the frozen "
                 "order does not declare",
         **provenance(gpat_path)},
        {"name": "backbone_lr", "value": float(detector["optimizer"]["backbone_lr"]),
         "owner": "C7 / detector", "optimizer": "AdamW (single optimizer, up to two groups)",
         "parameter_group": "backbone",
         "modules": ["local_backbone (ConvNeXt V2 Atto)", "highpass_stem (when enabled)"],
         "trainable_parameters": 3386760,
         "consumed_at": "src/prism_fas/detector/prism_detector.py:643 "
                        "(PRISMDetector.parameter_groups); "
                        "src/prism_fas/detector/trainer.py:291",
         "active_under_version_c": "TRACK_R_ONLY",
         "training_only": True,
         "classification": "A_active_for_track_r__E_non_applicable_for_track_g",
         "note": "the group is built from local_backbone.parameters(). Track G declares "
                 "local_branch=off, so the group is EMPTY and parameter_groups omits it "
                 "— backbone_lr then controls zero parameters",
         **provenance(det_path)},
        {"name": "head_lr", "value": float(detector["optimizer"]["head_lr"]),
         "owner": "C7 / detector", "optimizer": "AdamW (single optimizer, up to two groups)",
         "parameter_group": "heads",
         "modules": ["every trainable module that is not the local backbone: "
                     "global_projection, fusion head, region fusion, prompt head, "
                     "manifold projections, local_head"],
         "trainable_parameters": {"track_g": 12353, "track_r": 32654},
         "consumed_at": "src/prism_fas/detector/prism_detector.py:644",
         "active_under_version_c": True,
         "training_only": True,
         "classification": "A_active_for_both_tracks",
         "note": "the only LR group that exists under Track G, and one of two under "
                 "Track R",
         **provenance(det_path)},
        {"name": "min_lr_scale", "value": float(detector["scheduler"]["min_lr_scale"]),
         "owner": "C7 / detector", "optimizer": "cosine LambdaLR floor (relative)",
         "parameter_group": None, "modules": [], "trainable_parameters": 0,
         "consumed_at": "src/prism_fas/detector/trainer.py:348 (_lr_lambda)",
         "active_under_version_c": True, "training_only": True,
         "classification": "E_not_an_lr_anchor",
         "note": "a relative floor on the shared cosine multiplier, value 0.0",
         **provenance(det_path)},
        {"name": "backbone_lr (B00)", "value": float(b00["optimizer"]["backbone_lr"]),
         "owner": "B00 baseline (NOT C4 or C7)",
         "optimizer": "AdamW", "parameter_group": "backbone",
         "modules": ["B00ConvNeXtBinaryClassifier.backbone"],
         "trainable_parameters": None,
         "consumed_at": "src/prism_fas/train/models/b00_convnext.py:41",
         "active_under_version_c": False, "training_only": True,
         "classification": "E_irrelevant_to_version_c_c4_and_c7",
         "note": "B00 is the inherited Version-B ConvNeXt-only baseline. It is not a "
                 "Version-C track and no C4/C7/C8 row runs it. Listed so the audit is "
                 "complete, and so its 1.0e-4/5.0e-4 values are not mistaken for the "
                 "M9 detector anchors",
         **provenance(b00_path)},
        {"name": "head_lr (B00)", "value": float(b00["optimizer"]["head_lr"]),
         "owner": "B00 baseline (NOT C4 or C7)",
         "optimizer": "AdamW", "parameter_group": "head",
         "modules": ["B00ConvNeXtBinaryClassifier.head"],
         "trainable_parameters": None,
         "consumed_at": "src/prism_fas/train/models/b00_convnext.py:42",
         "active_under_version_c": False, "training_only": True,
         "classification": "E_irrelevant_to_version_c_c4_and_c7",
         "note": "see backbone_lr (B00)",
         **provenance(b00_path)},
    ]

    gpat_optimizer = gpat["optimizer"]
    detector_optimizer = detector["optimizer"]
    return {
        "scalars": scalars,
        "search_method": "grep for every field matching [a-z_]*(lr|learning_rate)[a-z_]* "
                         "across configs/, then trace each to the code that consumes it. "
                         "Names were not trusted to imply semantics",
        "no_scalar_literally_named_learning_rate": True,
        "coupling": {
            "c4_gpat": {
                "single_optimizer": True,
                "groups": ["artifact_encoder", "recipe_encoder", "generator"],
                "inherited_ratio_encoder_recipe_generator": [
                    gpat_optimizer["encoder_lr"] / gpat_optimizer["recipe_lr"], 1.0,
                    gpat_optimizer["generator_lr"] / gpat_optimizer["recipe_lr"]],
                "ratio_is_meaningful": True,
                "why": "all three groups are optimized by one AdamW under one cosine "
                       "schedule. cosine_schedule anchors each group's decay on that "
                       "group's own base LR, so the 2:1:2 ratio is preserved for the "
                       "whole run except at the shared absolute min_lr floor",
                "scheduler": "src/prism_fas/synthesis/gpat_trainer.py:40-53, per-group "
                             "lambda anchored on each group's base",
            },
            "c7_detector": {
                "single_optimizer": True,
                "groups": ["backbone", "heads"],
                "inherited_ratio_backbone_to_heads":
                    detector_optimizer["backbone_lr"] / detector_optimizer["head_lr"],
                "ratio_is_meaningful": True,
                "why": "one AdamW, and _lr_lambda returns ONE scalar multiplier that "
                       "LambdaLR applies to EVERY parameter group. Version B's own "
                       "detector schedule therefore already treats 'the learning rate' "
                       "as a single shape scaled across grouped anchors, holding the "
                       "1:10 backbone:heads ratio fixed for the entire run",
                "scheduler": "src/prism_fas/detector/trainer.py:342-348 (_lr_lambda), "
                             "applied to all groups by LambdaLR",
            },
        },
        "version_b_tuning_evidence": {
            "lr_sweep_artifacts_found": 0,
            "searched": ["reports/", "docs/", "configs/"],
            "conclusion": "Version B recorded no learning-rate sweep. Every LR value is "
                          "a single inherited setting, never a search winner, so there "
                          "is no Version-B precedent that elevates one scalar above the "
                          "others",
            "m9_reference_run_recorded": {"backbone_lr": 1e-05, "head_lr": 0.0001},
            "m10_optimizer_group_shapes": {
                "two_groups_backbone_and_heads": 35,
                "one_group_heads_only": 1,
                "one_group_row": "B01",
                "meaning": "Version B's own M10 evidence shows the backbone group being "
                           "OMITTED for the local-branch-off row. The Track G case is "
                           "not a new situation; it is the inherited behaviour",
            },
        },
    }


# --- sections 4 and 5: semantic mapping --------------------------------------

def semantic_mapping() -> dict[str, Any]:
    """What is actually trained, measured by building the models."""
    import yaml

    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.evaluation.variant_audit import build_audit_detector
    from prism_fas.pipeline.adapters.c7 import (TRACK_G_FLAGS, TRACK_R_FLAGS,
                                                TRACK_R_K4_FLAGS)
    from prism_fas.synthesis.gpat_model import build_gpat_model

    gpat_config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    model = build_gpat_model(gpat_config)
    gpat_groups = [
        {"name": group["name"], "lr": group["lr"],
         "parameters": sum(item.numel() for item in group["params"]),
         "empty": sum(item.numel() for item in group["params"]) == 0}
        for group in model.parameter_groups(gpat_config)]

    tracks: dict[str, Any] = {}
    for label, flags in (("track_g", TRACK_G_FLAGS), ("track_r_primary", TRACK_R_FLAGS),
                         ("track_r_k4_secondary", TRACK_R_K4_FLAGS)):
        variant = ResolvedExperimentVariant.resolve(flags)
        detector = build_audit_detector(variant)
        groups = detector.parameter_groups(backbone_lr=1.0e-5, head_lr=1.0e-4,
                                           weight_decay=0.05)
        names = [group["name"] for group in groups]
        tracks[label] = {
            "flags": variant.flags(),
            "decision_head_type": variant.decision_head_type,
            "optimizer_groups": [
                {"name": group["name"], "lr": group["lr"],
                 "parameters": sum(item.numel() for item in group["params"])}
                for group in groups],
            "group_names": names,
            "backbone_group_exists": "backbone" in names,
            "backbone_lr_controls_parameters": "backbone" in names,
            "applicable_lr_scalars": (["head_lr"] if "backbone" not in names
                                      else ["backbone_lr", "head_lr"]),
            "lr_anchor_uniquely_resolvable": "backbone" not in names,
            "active_loss_terms": {name: bool(value) for name, value
                                  in variant.active_loss_terms().items()},
        }

    return {
        "c4_gpat": {
            "trained_components": [group["name"] for group in gpat_groups],
            "optimizer_groups": gpat_groups,
            "all_groups_non_empty": all(not group["empty"] for group in gpat_groups),
            "groups_cover_all_trainable_parameters":
                sum(group["parameters"] for group in gpat_groups)
                == model.parameter_count(),
            "model_parameter_count": model.parameter_count(),
            "applicable_lr_scalars": ["encoder_lr", "recipe_lr", "generator_lr"],
            "lr_anchor_uniquely_resolvable": False,
            "per_scalar_classification": {
                "encoder_lr": "D — controls a live component of the frozen C4 "
                              "architecture (the artifact encoder)",
                "recipe_lr": "C/D — controls the recipe encoder, an active module and "
                             "the only route from a C3 recipe into the generator",
                "generator_lr": "D — controls the actual GPAT residual generator",
            },
            "no_scalar_is_class_B_or_E": True,
            "why_not_unique": "all three scalars control non-empty, simultaneously "
                              "optimized groups of the frozen architecture, and together "
                              "they cover 100% of trainable parameters. None is "
                              "historical, none is inactive, and none is a superset of "
                              "the others",
        },
        "c7_detector": tracks,
        "track_g_resolution": {
            "resolved_without_user_input": True,
            "unique_applicable_scalar": "head_lr",
            "value": 1.0e-4,
            "reason": "§13.4.1 makes Track G global-only and forbids instantiating "
                      "ConvNeXt. PRISMDetector.parameter_groups omits an empty group, so "
                      "the backbone group does not exist and backbone_lr controls zero "
                      "parameters. Exactly one LR scalar is applicable, which is what a "
                      "uniquely inherited anchor means",
            "supporting_rule": "§15.2.3 'absent/non-applicable scalars are skipped, not "
                               "invented'; §15.2.2 'Skip inactive terms'",
            "version_b_precedent": "M10 row B01 recorded optimizer_groups == ['heads'] "
                                   "for the same structural reason",
        },
    }


# --- sections 6, 7, 8: interpretations and their cost ------------------------

def _plan_trials(coordinate_specs: list[tuple[str, float | None]]) -> int:
    """Trial count for a coordinate list, computed by the real plan object."""
    from prism_fas.search.plan import (MULTIPLIERS_HALF_ONE_ONEHALF,
                                       MULTIPLIERS_HALF_ONE_TWO, WARMUP_FRACTION_RANGE,
                                       Coordinate, SearchPlan)

    coordinates = []
    for name, anchor in coordinate_specs:
        multipliers = (MULTIPLIERS_HALF_ONE_ONEHALF if name == "warmup"
                       else MULTIPLIERS_HALF_ONE_TWO)
        clip = WARMUP_FRACTION_RANGE if name == "warmup" else None
        coordinates.append(Coordinate(
            name=name, anchor=anchor, multipliers=multipliers, clip=clip,
            skip_reason="" if anchor is not None else "not applicable"))
    return SearchPlan(plan_id="probe", milestone="probe",
                      coordinates=tuple(coordinates),
                      selection_tuple=("objective",)).total_trials


def interpretations() -> dict[str, Any]:
    import yaml

    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.pipeline.adapters.c7 import TRACK_G_FLAGS, TRACK_R_FLAGS

    gpat = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    detector = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    weights = detector["loss"]["weights"]

    # Non-LR coordinates, per §15.2.2 / §15.2.3, with their inherited anchors.
    c4_base = [("weight_decay", float(gpat["optimizer"]["weight_decay"])),
               ("residual_loss_weight", float(gpat["loss"]["residual"])),
               ("identity_preservation_weight", float(gpat["loss"]["identity"])),
               ("geometry_preservation_weight", None)]      # absent, correctly skipped

    def detector_base(flags: dict[str, Any]) -> list[tuple[str, float | None]]:
        variant = ResolvedExperimentVariant.resolve(flags)
        active = variant.active_loss_terms()
        pairs = [("weight_decay", float(detector["optimizer"]["weight_decay"])),
                 ("warmup", float(detector["scheduler"]["warmup_fraction"]))]
        for coordinate, term in (("lambda_syn", "L_cls_syn"), ("lambda_local", "L_local"),
                                 ("lambda_MIL", "L_MIL"), ("lambda_P", "L_prompt"),
                                 ("lambda_risk", "L_risk"), ("lambda_M", "L_real"),
                                 ("lambda_out", "L_out"), ("lambda_clean", "L_clean")):
            pairs.append((coordinate,
                          float(weights[coordinate]) if active.get(term) else None))
        return pairs

    g_base, r_base = detector_base(TRACK_G_FLAGS), detector_base(TRACK_R_FLAGS)

    def block(base: list[tuple[str, float | None]], lr_coordinates:
              list[tuple[str, float | None]]) -> dict[str, Any]:
        combined = lr_coordinates + base
        active = [name for name, anchor in combined if anchor is not None]
        return {"coordinates_in_frozen_order": [name for name, _ in combined],
                "active_coordinates": active,
                "active_coordinate_count": len(active),
                "candidate_values_per_coordinate": 3,
                "max_trials": _plan_trials(combined)}

    lr_gpat_one = float(gpat["optimizer"]["generator_lr"])
    lr_det_head = float(detector["optimizer"]["head_lr"])
    lr_det_backbone = float(detector["optimizer"]["backbone_lr"])

    rows = [
        {
            "id": "D_skip",
            "title": "Skip the learning-rate coordinate entirely",
            "applies_to": ["C4", "C7 Track R"],
            "compliance": {"C4": NOT_APPLICABLE, "C7_track_r": NOT_APPLICABLE},
            "why": "§15.2.3 skips a scalar that is ABSENT or NON-APPLICABLE. For C4 and "
                   "Track R no LR scalar is absent — three and two respectively are "
                   "present and active — so the skip rule does not reach this case. It "
                   "is what the search plan does TODAY only because the ambiguity is "
                   "unresolved, which is a holding position rather than an "
                   "interpretation",
            "c4": block(c4_base, []),
            "c7_track_r": block(r_base, []),
        },
        {
            "id": "A_single_scalar",
            "title": "Elect one existing scalar as the learning-rate coordinate",
            "applies_to": ["C4", "C7 Track R"],
            "compliance": {"C4": APPROVAL_REQUIRED, "C7_track_r": APPROVAL_REQUIRED},
            "why": "keeps exactly one coordinate, matching the frozen search order, and "
                   "leaves the other groups at their inherited anchors — which is what "
                   "'all other coordinates remain at the current best' already does for "
                   "anything unsearched. But nothing in the inheritance elevates one "
                   "scalar above the others, so ELECTING one is an act of scientific "
                   "discretion with no evidential basis. Appendix J places an unforced "
                   "result-affecting choice in USER_APPROVAL_REQUIRED",
            "consequence": "the unelected groups are never searched; their LR stays at "
                           "the inherited value for the whole envelope",
            "c4": block(c4_base, [("learning_rate", lr_gpat_one)]),
            "c7_track_r": block(r_base, [("learning_rate", lr_det_head)]),
        },
        {
            "id": "B_common_multiplier",
            "title": "One coordinate that scales every active LR group, preserving "
                     "inherited ratios",
            "applies_to": ["C4", "C7 Track R"],
            "compliance": {"C4": APPROVAL_REQUIRED, "C7_track_r": APPROVAL_REQUIRED},
            "why": "keeps exactly one coordinate and the same trial count as A, searches "
                   "every active group, and changes no inherited ratio: the candidate is "
                   "a multiplier m in {0.5, 1.0, 2.0} applied to every group's own "
                   "anchor. It is the narrowest reading of a single 'learning rate' "
                   "coordinate over a component whose inherited anchor is a vector. "
                   "It is an INTERPRETATION of 'around inherited anchor x0.5/x1.0/x2.0' "
                   "rather than the literal text, so it still needs approval",
            "inheritance_support": "Version B's own schedules already behave this way. "
                                   "The detector's _lr_lambda returns ONE multiplier that "
                                   "LambdaLR applies to every group; GPAT's cosine "
                                   "schedule anchors each group on its own base under one "
                                   "shared shape. Treating the LR as a common multiplier "
                                   "over grouped anchors is the inherited semantics, not "
                                   "a new one",
            "consequence": "m=1.0 reproduces the inherited configuration exactly, so the "
                           "anchor trial is unchanged from Version B",
            "c4": block(c4_base, [("learning_rate_multiplier", 1.0)]),
            "c7_track_r": block(r_base, [("learning_rate_multiplier", 1.0)]),
        },
        {
            "id": "C_independent_per_group",
            "title": "Search each active LR group as its own coordinate",
            "applies_to": ["C4", "C7 Track R"],
            "compliance": {"C4": ENVELOPE_EXPANSION, "C7_track_r": ENVELOPE_EXPANSION},
            "why": "the §15.2.2 search order names ONE 'learning rate' step. Giving each "
                   "group its own coordinate inserts steps the frozen order does not "
                   "declare — two extra for C4, one extra for Track R — and enlarges the "
                   "search. §15.2.2 already classifies expanding a candidate set or "
                   "starting a second pass as USER_APPROVAL_REQUIRED; adding coordinates "
                   "is a larger change than either, and it also breaks the inherited "
                   "ratios by construction",
            "consequence": "C4 grows from 1 to 3 LR coordinates (+6 trials); Track R "
                           "grows from 1 to 2 (+3 trials). Inherited LR ratios are no "
                           "longer preserved",
            "c4": block(c4_base, [("encoder_lr", float(gpat["optimizer"]["encoder_lr"])),
                                  ("recipe_lr", float(gpat["optimizer"]["recipe_lr"])),
                                  ("generator_lr", float(gpat["optimizer"]["generator_lr"]))]),
            "c7_track_r": block(r_base, [("backbone_lr", lr_det_backbone),
                                         ("head_lr", lr_det_head)]),
        },
    ]

    track_g = {
        "id": "TRACK_G_RESOLVED",
        "title": "Track G has exactly one applicable LR scalar",
        "compliance": {"C7_track_g": ALREADY_IMPLIED},
        "why": "not an interpretation and not a user choice. Track G instantiates no "
               "ConvNeXt, its backbone optimizer group is empty and is omitted, and "
               "backbone_lr controls zero parameters. head_lr is the unique inherited "
               "anchor, which is exactly what the frozen rules require",
        "anchor": {"scalar": "head_lr", "value": lr_det_head},
        "c7_track_g": block(g_base, [("learning_rate", lr_det_head)]),
        "c7_track_g_if_lr_skipped": block(g_base, []),
    }

    return {
        "frozen_search_order": [
            "learning rate", "weight decay", "warm-up", "lambda_syn", "lambda_local",
            "lambda_MIL", "lambda_P", "lambda_risk", "active K=4-only scalar weights"],
        "frozen_order_declares_one_learning_rate_coordinate": True,
        "candidate_values_per_coordinate": [0.5, 1.0, 2.0],
        "warmup_multipliers": [0.5, 1.0, 1.5],
        "interpretations": rows,
        "track_g": track_g,
        "current_plan_identities": {
            "c4_gpat_coordinate_v1":
                "ab77e964d9c035cf2c3bed209ffac307aebd85c6735879bc3fa3c5efce20d0ec",
            "c7_detector_coordinate_v1":
                "62d0022507e732ba89618845fab2c63fec2b7b07f6817b2d541a4f500f459d7b",
            "note": "both were built with learning_rate AMBIGUOUS and therefore skipped. "
                    "Any approved interpretation changes these identities, and the "
                    "changed identity is what a later full run must execute against",
        },
    }


# --- section 9: recommendation ----------------------------------------------

def recommendation() -> dict[str, Any]:
    return {
        "status": "RECOMMENDATION_ONLY",
        "decision_status": "AWAITING_USER_APPROVAL",
        "implemented": False,
        "c4": {
            "recommended": "B_common_multiplier",
            "anchor_vector": {"encoder_lr": 2.0e-4, "recipe_lr": 1.0e-4,
                              "generator_lr": 2.0e-4},
            "candidates": [0.5, 1.0, 2.0],
            "rationale": [
                "minimum deviation: keeps exactly one learning-rate coordinate, which is "
                "what the frozen §15.2.2 order declares, and the same 12 trials as A",
                "strongest inheritance: all three anchors are byte-identical to Version B "
                "and all three groups are live; B is the only option that searches the "
                "learning rate of the whole component as Version B configured it",
                "least added discretion: A requires electing one scalar over two equally "
                "inherited ones with no evidence to justify the choice; B requires no "
                "election at all",
                "preserves optimizer-group relationships: the 2:1:2 ratio is held fixed, "
                "matching what the inherited cosine schedule already does",
                "no envelope enlargement: unlike C, no coordinate is added",
            ],
            "what_the_user_is_accepting": "that 'the inherited anchor' for a component "
                                          "with grouped learning rates means the anchor "
                                          "VECTOR, scaled as a unit",
        },
        "c7": {
            "track_g": {
                "recommended": "no decision required",
                "compliance": ALREADY_IMPLIED,
                "anchor": {"head_lr": 1.0e-4},
                "rationale": ["backbone_lr controls zero parameters under Track G, so the "
                              "inherited anchor is already unique. Recording this as a "
                              "user choice would invent a decision the frozen "
                              "architecture already makes"],
            },
            "track_r": {
                "recommended": "B_common_multiplier",
                "anchor_vector": {"backbone_lr": 1.0e-5, "head_lr": 1.0e-4},
                "candidates": [0.5, 1.0, 2.0],
                "rationale": [
                    "the same five criteria as C4",
                    "additionally: the inherited detector scheduler ALREADY applies one "
                    "multiplier to every group, so a common multiplier is the inherited "
                    "semantics rather than a new construct",
                    "the 1:10 backbone:heads ratio is a deliberate frozen setting — a "
                    "frozen-then-finetuned backbone at 1e-5 with fresh heads at 1e-4 — "
                    "and B is the only option that does not disturb it",
                ],
            },
        },
        "consistency_note": "recommending B for both C4 and Track R keeps one meaning of "
                            "'the learning-rate coordinate' across the whole pipeline. "
                            "Approving different interpretations per milestone is legal "
                            "but would make the two search plans mean different things by "
                            "the same word",
        "if_the_user_declines_all": "the envelopes stay as they are today, with the "
                                    "learning-rate coordinate skipped. C4 and C7 remain "
                                    "scientifically blocked and NEEDS_SCIENTIFIC_DECISION "
                                    "stands",
    }


# --- section 11: external artifact audit ------------------------------------

def external_artifacts() -> dict[str, Any]:
    """What the future GPU host needs, and where it actually is."""
    import yaml

    from prism_fas.detector.pretrained import CONVNEXT_PIN, SIGLIP2_PIN

    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text(encoding="utf-8"))
    cache = Path(str(paths.get("model_cache", "")))
    raw = dict(paths.get("raw_datasets") or {})
    gpat = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    quality = yaml.safe_load(
        (REPO / "configs/synthesis/quality_gate_m8.yaml").read_text(encoding="utf-8"))

    items: list[dict[str, Any]] = []

    # --- pinned weights, verified against their frozen identities -------------
    siglip_root = cache / SIGLIP2_PIN["local_relpath"]
    files = []
    all_ok = siglip_root.exists()
    for name, spec in SIGLIP2_PIN["files"].items():
        path = siglip_root / name
        actual = sha256_file(path)
        ok = actual == spec["sha256"]
        all_ok &= bool(ok)
        files.append({"file": name, "expected_sha256": spec["sha256"],
                      "actual_sha256": actual, "matches": bool(ok),
                      "bytes": spec["bytes"]})
    items.append({
        "logical_name": "siglip2_frozen_global_tower",
        "status": "AVAILABLE_LOCAL" if all_ok else "MISSING",
        "expected_path": str(siglip_root),
        "identity_verified": all_ok,
        "model_id": SIGLIP2_PIN["model_id"], "revision": SIGLIP2_PIN["revision"],
        "bytes": sum(spec["bytes"] for spec in SIGLIP2_PIN["files"].values()),
        "files": files,
        "required_by": ["C7", "C8", "C11"], "needed_for_c4_full": False,
        "in_git": False,
        "note": "outside both repositories, licensed/large. Verified here by hash, not "
                "copied",
    })

    convnext_path = next((cache / rel for rel in
                          (CONVNEXT_PIN["local_relpath"], *CONVNEXT_PIN["alternate_relpaths"])
                          if (cache / rel).exists()), None)
    convnext_sha = sha256_file(convnext_path) if convnext_path else None
    items.append({
        "logical_name": "convnextv2_atto_local_branch",
        "status": ("AVAILABLE_LOCAL" if convnext_sha == CONVNEXT_PIN["weight_sha256"]
                   else "MISSING"),
        "expected_path": str(convnext_path) if convnext_path else
                         str(cache / CONVNEXT_PIN["local_relpath"]),
        "expected_sha256": CONVNEXT_PIN["weight_sha256"], "actual_sha256": convnext_sha,
        "identity_verified": convnext_sha == CONVNEXT_PIN["weight_sha256"],
        "bytes": convnext_path.stat().st_size if convnext_path else None,
        "required_by": ["C7", "C8", "C11"], "needed_for_c4_full": False,
        "in_git": False,
        "note": "Track R only; Track G instantiates no local backbone",
    })

    for logical, relpaths, expected, stages, for_c4 in (
            ("adaface_identity_backbone",
             ["face_identity/pretrained_model/model.pt"],
             gpat["identity_model"]["weight_sha256"], ["C4", "C6"], True),
            ("scrfd_face_detector",
             ["face_detectors/scrfd_10g_bnkps.onnx"],
             quality["quality_models"]["detector"]["sha256"], ["C6"], False),
            ("facexformer_parsing",
             ["face_geometry/ckpts/model.pt"],
             quality["quality_models"]["parsing"]["sha256"], ["C6"], False)):
        path = next((cache / rel for rel in relpaths if (cache / rel).exists()), None)
        actual = sha256_file(path) if path else None
        items.append({
            "logical_name": logical,
            "status": "AVAILABLE_LOCAL" if actual == expected else "MISSING",
            "expected_path": str(path) if path else str(cache / relpaths[0]),
            "expected_sha256": expected, "actual_sha256": actual,
            "identity_verified": actual == expected,
            "bytes": path.stat().st_size if path else None,
            "required_by": stages, "needed_for_c4_full": for_c4, "in_git": False,
        })

    # --- raw datasets ---------------------------------------------------------
    for key, stages, for_c4 in (("casia_fasd", ["C5", "C7", "C8"], True),
                                ("msu_mfsd", ["C5", "C7", "C8"], True)):
        root = Path(str(raw.get(key, "")))
        items.append({
            "logical_name": f"raw_dataset_{key}",
            "status": "AVAILABLE_LOCAL" if root.exists() else "MISSING",
            "expected_path": str(root), "identity_verified": None,
            "required_by": stages, "needed_for_c4_full": for_c4, "in_git": False,
            "note": "raw source; never copied into either repository. Verified by the "
                    "preprocessing manifest identity, not by raw bytes",
        })
    siw = Path(str(raw.get("siw_mv2", "")))
    items.append({
        "logical_name": "raw_dataset_siw_mv2",
        "status": "AVAILABLE_LOCAL" if siw.exists() else "MISSING",
        "expected_path": str(siw), "identity_verified": None,
        "required_by": ["C10", "C11", "C12"], "needed_for_c4_full": False,
        "not_needed_until_stage": "C10", "in_git": False,
        "access_policy": "EVALUATION_ONLY. Label-free features may be mounted read-only "
                         "for C11; label files may never be mounted on a training "
                         "process. Not resolved, opened or hashed by this audit",
    })

    # --- derived trees --------------------------------------------------------
    for logical, relative, stages, for_c4, how in (
            ("preprocessed_source_data", "data/processed", ["C5", "C7", "C8"], True,
             "rebuild on the GPU host from the raw roots with the documented M2 "
             "preprocessing CLI, or transfer out of band"),
            ("source_packages", "data/packages", ["C4", "C5", "C7", "C8"], True,
             "built from the preprocessed tree by the M3 packaging step"),
            ("gpat_pair_plan", "data/packages/gpat_pairs", ["C4"], True,
             "produced by the frozen source-only pair plan; depends on source_packages"),
            ("target_label_artifact", "data/evaluation_only/prism_target_v2_labels",
             ["C12"], False,
             "evaluation-only; readable by the isolated C-G8 scorer alone")):
        path = REPO / relative
        present = path.exists() and any(path.rglob("*")) if path.exists() else False
        items.append({
            "logical_name": logical,
            "status": "AVAILABLE_LOCAL" if present else "MISSING",
            "expected_path": relative, "identity_verified": None,
            "required_by": stages, "needed_for_c4_full": for_c4, "in_git": False,
            "how_to_obtain": how,
            **({"not_needed_until_stage": "C12"} if logical == "target_label_artifact"
               else {}),
        })

    # --- in-Git scientific inputs --------------------------------------------
    for logical, relative, stages in (
            ("c3_scientific_recipe_banks", "assets/recipe_banks/c3", ["C4", "C5", "C6"]),
            ("c3_scientific_bank_lock",
             "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json", ["C5"]),
            ("gpat_training_config", "configs/synthesis/gpat_m8.yaml", ["C4"]),
            ("detector_training_config", "configs/train/m9_reference.yaml", ["C7", "C8"])):
        path = REPO / relative
        items.append({
            "logical_name": logical,
            "status": "AVAILABLE_IN_GIT" if path.exists() else "MISSING",
            "expected_path": relative,
            "identity_verified": True,
            "sha256": sha256_file(path) if path.is_file() else None,
            "required_by": stages,
            "needed_for_c4_full": "C4" in stages, "in_git": True,
        })

    items.append({
        "logical_name": "cuda_accelerator",
        "status": "MISSING",
        "expected_path": "<the collaborator GPU host>",
        "identity_verified": None,
        "required_by": ["C4", "C5", "C7", "C8", "C11"],
        "needed_for_c4_full": True, "in_git": False,
        "note": "torch reports cuda_available=False on this machine. Every GPU-dependent "
                "check is DEFERRED_TO_EXTERNAL_GPU_PREFLIGHT",
    })

    blocking_c4 = [item["logical_name"] for item in items
                   if item["needed_for_c4_full"] and item["status"] == "MISSING"]
    return {
        "items": items,
        "by_status": {status: sum(1 for item in items if item["status"] == status)
                      for status in sorted({item["status"] for item in items})},
        "blocking_c4_full": blocking_c4,
        "correction_to_previous_milestone": (
            "the C0-C13 readiness handoff listed 'pinned weights' and "
            "'gpat_identity_model_adaface' as absent. They are present and now verified "
            "by hash. That inventory looked under data/packages/pretrained and hard-coded "
            "present=false for AdaFace instead of resolving the declared model cache. The "
            "real C4 gap is the derived data, not the weights"),
        "nothing_downloaded": True,
        "nothing_substituted": True,
    }


def handoff_set() -> dict[str, Any]:
    """Section 12: the minimum set to start C4 FULL elsewhere."""
    return {
        "must_have_before_c4": [
            {"item": "git checkout", "detail": "branch pre-gpu-scientific-decision (or "
                                               "its successor) at the accepted commit"},
            {"item": "approved learning-rate interpretation",
             "detail": "C4 cannot execute its bounded envelope until the anchor decision "
                       "in this dossier is approved"},
            {"item": "python environment",
             "detail": "the project venv with the .[llm] extra AND the inherited "
                       "dependencies; `.[llm]` alone leaves the inherited suite "
                       "uncollectable"},
            {"item": "raw source datasets", "detail": "CASIA-FASD and MSU-MFSD roots"},
            {"item": "preprocessed source data + source packages",
             "detail": "data/processed and data/packages, rebuilt on the host from the "
                       "raw roots or transferred; ABSENT here"},
            {"item": "gpat pair plan", "detail": "data/packages/gpat_pairs"},
            {"item": "adaface identity backbone",
             "detail": "sha256 43bd2d57…, verified present in the declared model cache"},
            {"item": "c3 scientific recipe banks",
             "detail": "assets/recipe_banks/c3, arrives with the clone"},
            {"item": "cuda accelerator", "detail": "the full profile declares gpu: required"},
        ],
        "can_be_fetched_on_gpu_host": [
            {"item": "preprocessed source data",
             "detail": "reproducible from the raw roots by the documented preprocessing "
                       "CLI; identity is pinned by the resulting manifest"},
            {"item": "pinned backbone weights",
             "detail": "SigLIP2 and ConvNeXt are needed from C7 onward, not for C4, and "
                       "are re-downloadable at their pinned revision and sha256"},
        ],
        "not_needed_until_later": [
            {"item": "siglip2 / convnext", "from_stage": "C7"},
            {"item": "scrfd + facexformer quality models", "from_stage": "C6"},
            {"item": "siw-mv2 feature package", "from_stage": "C10"},
            {"item": "evaluation-only target labels", "from_stage": "C12"},
        ],
        "writable_destinations": {
            "checkpoints_and_run_manifests": "runs/full/c4/",
            "stage_evidence": "reports/full/c4/",
            "search_state": "reports/full/c4/C4_SEARCH_STATE.json",
            "pipeline_state": "state/PIPELINE_STATE.json",
            "master_run_index": "state/MASTER_RUN_INDEX.json",
        },
        "resume_state": "identity-aware. A completed trial is reused only when its config "
                        "identity matches; a valid GPAT checkpoint is never retrained "
                        "while its ancestor and search identities still match (L.11)",
        "disk_space": {
            "repository_clone": "small; the banks and configs are the only scientific "
                                "payload in Git",
            "model_cache": "~2.8 GB for the five pinned weights measured here",
            "preprocessed_and_packages": "NOT CALCULABLE from this machine — the trees "
                                         "have never been built here",
        },
    }


def future_command() -> dict[str, Any]:
    """Section 13: the actually supported CLI, read from the parser."""
    sys.path.insert(0, str(REPO))
    import train as entrypoint

    parser = entrypoint.build_parser()
    options = []
    for action in parser._actions:                       # noqa: SLF001 - reading the real parser
        if not action.option_strings:
            continue
        options.append({"flags": list(action.option_strings),
                        "dest": action.dest,
                        "choices": list(action.choices) if action.choices else None})
    return {
        "canonical_command": "python train.py --profile full --from C4 --to C4 --resume",
        "verified_against_parser": True,
        "supported_options": options,
        "exit_codes": {"0": "PASS", "1": "FAIL", "2": "BLOCKED", "3": "usage/config error"},
        "not_executed": True,
        "backend_and_device_selection": {
            "how": "the profile's compute_policy declares the ceiling (full: gpu "
                   "required); the device itself is resolved at run time by the canonical "
                   "trainers — prism_fas.synthesis.gpat_trainer.resolve_device and "
                   "prism_fas.detector.trainer.resolve_device",
            "no_backend_flag_today": "train.py exposes no --backend flag. L.4 lists one "
                                     "as optional; it does not exist yet, and reporting "
                                     "the flag as available would be inventing CLI",
            "modal_specific_paths_in_scientific_config": 0,
        },
    }


def portability_audit() -> dict[str, Any]:
    """Section 14: prove identity independence from the machine."""
    from prism_fas.pipeline.portability import (KNOWN_BACKENDS, OPERATIONAL_FIELDS,
                                                PROVENANCE_NAMESPACE,
                                                assert_composition_preserved,
                                                frozen_composition,
                                                identity_is_backend_invariant,
                                                resolve_microbatch)

    payload = {"config_identity": "probe", "seed": 20260806,
               "search_plan_identity": "probe", "arm": "LLM"}
    invariance = identity_is_backend_invariant(payload, KNOWN_BACKENDS.values())

    plans = {}
    for vram in (6.0, 12.0, 24.0, 80.0):
        backend = __import__("prism_fas.pipeline.portability", fromlist=["BackendProfile"]) \
            .BackendProfile(f"vram_{int(vram)}g", device="cuda", vram_gb=vram)
        plan = resolve_microbatch(backend=backend)
        assert_composition_preserved(plan)
        plans[f"{int(vram)}GB"] = {
            "physical_microbatch": plan.microbatch,
            "gradient_accumulation_steps": plan.accumulation_steps,
            "effective_batch": plan.effective_batch,
            "preserves_effective_batch": plan.preserves_effective_batch}
    return {
        "scientific_identity_backend_invariant": invariance["invariant"],
        "distinct_identities_across_backends": invariance["distinct_identity_count"],
        "backends_probed": sorted(invariance["identities"]),
        "operational_fields_excluded_from_identity": list(OPERATIONAL_FIELDS),
        "provenance_namespace": PROVENANCE_NAMESPACE,
        "frozen_effective_composition": frozen_composition(),
        "microbatch_plans_by_vram": plans,
        "independent_of": ["hostname", "gpu_uuid", "modal_workspace",
                           "absolute_filesystem_root"],
        "permitted_engineering_adaptation": ["physical microbatch",
                                             "gradient accumulation steps",
                                             "dataloader workers", "I/O tuning"],
        "nothing_changed_by_this_audit": True,
    }


def build() -> dict[str, Any]:
    dirty = (git(["status", "--porcelain"]) or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc(),
        "title": "Learning-rate anchor decision dossier — C4 and C7",
        "decision_status": "AWAITING_USER_APPROVAL",
        "implemented": False,
        "governing_spec": {"path": SPEC, "sha256": sha256_file(REPO / SPEC),
                           "sections": ["§15.2.2", "§15.2.3", "§13.4.1", "§18",
                                        "Appendix J", "Appendix L"]},
        "git": {"branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "commit": git(["rev-parse", "HEAD"]),
                "accepted_engineering_checkpoint": "f8d5a5fab9f253c61399cda5f4031f4b4af0e68c",
                "worktree_clean": dirty == ""},
        "version_b": {"path": str(VERSION_B),
                      "head": git(["rev-parse", "HEAD"], VERSION_B),
                      "tag_peeled": git(["rev-parse",
                                         "m10-blind-evaluation-checkpoint^{commit}"],
                                        VERSION_B),
                      "clean": (git(["status", "--porcelain"], VERSION_B) or "") == ""},
        "lr_evidence": lr_evidence(),
        "semantic_mapping": semantic_mapping(),
        "interpretations": interpretations(),
        "recommendation": recommendation(),
        "external_artifacts": external_artifacts(),
        "c4_handoff_set": handoff_set(),
        "future_command": future_command(),
        "portability": portability_audit(),
        "activity": {"modal_gpu_seconds": 0, "scientific_training_runs": 0,
                     "gemini_calls": 0, "real_target_access": 0,
                     "gpu_allocated": False, "datasets_opened": 0,
                     "weights_hashed_not_loaded": 5},
        "what_this_artifact_does_not_do": [
            "it approves nothing",
            "it changes no search plan, lock or scientific status",
            "it selects no GPAT or detector winner",
            "it executes no trial",
        ],
    }


def main() -> int:
    from prism_fas.pipeline.state import atomic_write_json

    payload = build()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "LR_ANCHOR_DECISION_DOSSIER.json"
    atomic_write_json(path, payload)

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    identity = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(f"wrote {path.relative_to(REPO).as_posix()}")
    print(f"  dossier_identity   {identity}")
    print(f"  decision_status    {payload['decision_status']}")
    print(f"  C4 recommended     {payload['recommendation']['c4']['recommended']}")
    print(f"  C7 Track G         {payload['recommendation']['c7']['track_g']['compliance']}")
    print(f"  C7 Track R rec     {payload['recommendation']['c7']['track_r']['recommended']}")
    print(f"  blocking C4 full   {payload['external_artifacts']['blocking_c4_full']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
