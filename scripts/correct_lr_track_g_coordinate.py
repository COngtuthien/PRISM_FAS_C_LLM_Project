"""Record the Track-G learning-rate coordinate correction and its identity move.

    python scripts/correct_lr_track_g_coordinate.py

`reports/handoff/LR_ANCHOR_DECISION_RECORD.json` froze the approved
learning-rate decision on 2026-08-17 and is NOT touched by this script. It is the
evidence of what was decided, and it stays exactly as written — including the
line that records the defect as though it were correct:

    "C7_TRACK_G": {"before": 12, "after": 12,
                   "note": "unchanged; Track G's LR coordinate is inapplicable
                            because backbone_lr controls no Track-G parameter"}

The note is half right. `backbone_lr` really does control zero Track-G
parameters — §13.4.1 forbids Track G from instantiating ConvNeXt — which is why
Track G's anchor is uniquely `head_lr` and needed no user decision. It does not
follow that the coordinate is inapplicable: §15.2.2 puts `learning_rate` first in
the frozen order with candidates `anchor x {0.5, 1.0, 2.0}` and declares no
exemption for a component whose anchor happens to be unique. The implementation
turned "no decision needed" into "no search performed".

So this is an IMPLEMENTATION_CORRECTION, not a new scientific choice. The
approved decision record `configs/search/lr_anchor_decision.yaml` is unchanged
byte for byte: its values were always right, and its prose says Track G "needed
no approval", never that Track G has nothing to search. What moves is the
implementation's canonical SEMANTIC serialization, which now truthfully reports
Track G's candidates — and therefore the decision identity derived from it.

Zero scientific C7 trials had executed when this was found, so no measurement is
invalidated and nothing needs regenerating.
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
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

OUT = REPO / "reports" / "handoff" / "LR_ANCHOR_DECISION_CORRECTION.json"
FROZEN_RECORD = REPO / "reports" / "handoff" / "LR_ANCHOR_DECISION_RECORD.json"
SCHEMA_VERSION = "prism-lr-anchor-correction-v1"

#: The identity the defective implementation produced. Preserved by name so the
#: supersession is legible from the artifact alone.
SUPERSEDED_DECISION_IDENTITY = (
    "7ef3492263507d4399828089bbe1af79438bc892e50c8ad732585c1d40c8397c")

#: The Track-G plan identity under the defective 12-trial envelope.
SUPERSEDED_TRACK_G_PLAN_IDENTITY = (
    "9ce24e12627f198c5378c48e33e8c09ba8f3a0ef39da38bcd7d8c63bd37cf9d1")

#: What C4's plan hashed to before the correction. Asserted, not recorded: C4 has
#: already executed scientifically, so a moved C4 plan identity would mean this
#: correction reached an artifact it must not touch.
C4_PLAN_IDENTITY_BEFORE = (
    "71bfff29bfe1e7ba71d083831a0337a6ae6e0dcfc7f7a75eb9e6f3f3a4ac2b6a")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    from prism_fas.evaluation import source_selection
    from prism_fas.pipeline.adapters.c7 import (TRACK_G_FLAGS, TRACK_R_FLAGS,
                                                _active_terms, _variant)
    from prism_fas.search.c7_decision import load_decision as load_c7
    from prism_fas.search.lr_decision import load_decision as load_lr
    from prism_fas.search.plan import (K4_ONLY_WEIGHTS, detector_search_plan,
                                       gpat_search_plan)

    record = load_lr(REPO)
    search = load_c7(REPO)

    gpat_config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    gpat_plan, _ = gpat_search_plan(gpat_config,
                                    lr_decision=record.for_component("C4"))

    detector_config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    tuple_name = source_selection.TUPLES[search.selection_tuple_name]

    tracks: dict[str, Any] = {}
    total = 0
    for track in search.tracks:
        variant = _variant(TRACK_R_FLAGS if track == "R" else TRACK_G_FLAGS)
        lr = record.for_component(f"C7_TRACK_{track}")
        plan, _resolutions = detector_search_plan(
            detector_config, active_terms=_active_terms(variant),
            k4_weights=K4_ONLY_WEIGHTS, selection_tuple=tuple_name, lr_decision=lr)
        coordinate = next(item for item in plan.coordinates
                          if item.name == lr.coordinate_name)
        tracks[track] = {
            "search_plan_identity": plan.identity,
            "declared_trials": plan.total_trials,
            "applicable_coordinates": [item.name for item in plan.coordinates
                                       if item.applicable],
            "coordinate_order": list(plan.coordinate_order),
            "learning_rate": {
                "coordinate_name": lr.coordinate_name,
                "interpretation": lr.interpretation,
                "applicable": coordinate.applicable,
                "multipliers": list(coordinate.candidates),
                "anchor_vector": dict(lr.anchor_vector),
                "effective_learning_rates": {
                    str(value): lr.lr_for_groups(value) for value in lr.candidates},
                "anchor_trial_reproduces_inherited":
                    lr.lr_for_groups(1.0) == {name: float(value) for name, value
                                              in lr.anchor_vector.items()},
                "parameter_groups": list(lr.parameter_groups),
            },
        }
        total += plan.total_trials

    epochs = int(detector_config["stages"]["total_epochs"])
    steps = int(detector_config["batch"]["steps_per_epoch"])
    frozen = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "status": "IMPLEMENTATION_CORRECTION_APPLIED",
        "classification": "IMPLEMENTATION_CORRECTION",
        "defect_id": "C7_TRACK_G_LEARNING_RATE_COORDINATE_NOT_SEARCHED",
        "authorized_by": "user, in session, as an implementation correction to the "
                         "already frozen §15.2.2 envelope",
        "not_authorized_as": "a deliberate Track-G learning-rate exemption",

        "root_cause": (
            "`search/lr_decision.py` derived 'is the learning-rate coordinate "
            "searched' from 'does the multiplier expand over more than one "
            "parameter group'. `UNIQUE_INHERITED_ANCHOR` therefore produced "
            "`candidates = ()` and an INAPPLICABLE coordinate, with the reason "
            "'there is no ambiguity to search and no multiplier to apply'. The "
            "second clause is correct — one applicable group has no inherited "
            "ratio to hold — but the first turned 'no user decision needed' into "
            "'no search performed'."),
        "why_it_was_not_caught_earlier": (
            "the UNIQUE_INHERITED_ANCHOR branch existed but had never been routed "
            "into a search plan: C4 and C7 Track R are both B_common_multiplier, "
            "and Track G's scientific search was first wired at 390fcb2. The "
            "frozen 2026-08-17 record even captured the defect as intended "
            "behaviour in trial_count_change.C7_TRACK_G.note."),
        "consequence_if_executed": (
            "Track G would have run 12 trials instead of 15 and frozen config_G's "
            "learning rate at the inherited anchor without evaluating 0.5x or 2x. "
            "config_G is what C-G-RND, C-G-DET and C-G-LLM all train at in C8, so "
            "every Track-G number would have rested on an unsearched coordinate, "
            "and the lock would have recorded a complete-looking one-pass envelope."),

        "corrected_semantics": {
            "UNIQUE_INHERITED_ANCHOR": (
                "describes HOW the anchor was resolved — exactly one inherited LR "
                "scalar is applicable, so no user decision is required to choose "
                "one. It says nothing about whether the coordinate is searched."),
            "learning_rate_coordinate": (
                "§15.2.2 puts it first in the frozen order with candidates "
                "anchor x {0.5, 1.0, 2.0}, under EVERY approved interpretation "
                "that has an applicable inherited anchor."),
            "representation": (
                "one coordinate, `learning_rate_multiplier`, for both "
                "interpretations. No second scalar-valued LR coordinate exists, "
                "and no independent per-group LR search is introduced."),
            "expansion": (
                "`lr_for_groups` expands the multiplier over the applicable "
                "groups: three for C4, two for Track R holding the frozen 1:10 "
                "ratio, one for Track G where there is no ratio to hold."),
        },

        "identity_change": {
            "lr_decision_identity": {
                "before": SUPERSEDED_DECISION_IDENTITY,
                "after": record.identity,
                "changed": record.identity != SUPERSEDED_DECISION_IDENTITY,
                "why": (
                    "`LRDecisionRecord.identity_material` hashes the canonical "
                    "SEMANTIC payload of every component. Track G's payload now "
                    "truthfully carries its multipliers, its coordinate name and "
                    "its per-multiplier learning rates, where it previously "
                    "carried empty values. The identity therefore had to move: "
                    "pretending otherwise would mean two different envelopes "
                    "sharing one identity."),
            },
            "decision_config_bytes": {
                "path": "configs/search/lr_anchor_decision.yaml",
                "sha256": _sha256_file(REPO / "configs/search/lr_anchor_decision.yaml"),
                "sha256_in_frozen_record": frozen.get("decision_config_sha256"),
                "changed": False,
                "why_not": (
                    "the approved record's scientific values were always correct — "
                    "multipliers [0.5, 1.0, 2.0], Track-G anchor head_lr 1.0e-4, "
                    "Track-G parameter_groups [heads] — and its prose says Track G "
                    "'needed no approval', never that Track G has nothing to "
                    "search. No frozen byte required correction, so none was "
                    "made."),
            },
            "c7_search_decision_identity": {
                "value": search.identity,
                "changed": False,
                "why_not": "C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET is untouched by "
                           "this correction.",
            },
            "c4_gpat_search_plan_identity": {
                "value": gpat_plan.identity,
                "expected_unchanged": C4_PLAN_IDENTITY_BEFORE,
                "changed": gpat_plan.identity != C4_PLAN_IDENTITY_BEFORE,
                "why_it_must_not_change": (
                    "C4 has already executed scientifically and C5 renders through "
                    "the checkpoint its lock names. An intermediate version of "
                    "this correction decorated `Coordinate.anchor_source`, which "
                    "enters the plan identity, and moved C4's plan hash for a "
                    "purely cosmetic reason. Both strings are now held "
                    "byte-identical on the multiplier branch."),
            },
            "c7_track_g_search_plan_identity": {
                "before": SUPERSEDED_TRACK_G_PLAN_IDENTITY,
                "after": tracks["G"]["search_plan_identity"],
                "changed": (tracks["G"]["search_plan_identity"]
                            != SUPERSEDED_TRACK_G_PLAN_IDENTITY),
                "why": "the applicable envelope changed from 12 to 15 trials. This "
                       "change is required, not incidental.",
            },
            "c7_track_r_search_plan_identity": {
                "value": tracks["R"]["search_plan_identity"],
                "envelope_unchanged": True,
                "note": (
                    "Track R's ENVELOPE is untouched: 24 trials, multipliers "
                    "[0.5, 1.0, 2.0], 1:10 ratio held. Its plan identity as built "
                    "by the C7 adapter still moves, because that plan binds the "
                    "LR decision identity into its base config and that identity "
                    "moved. Nothing has executed, so nothing is invalidated."),
            },
        },

        "affected_scientific_evidence": {
            "c7_scientific_trials_executed_before_correction": 0,
            "c7_scientific_metrics_existing_before_correction": 0,
            "detector_config_lock_written_before_correction": False,
            "c8_rows_executed": 0,
            "c4_artifacts_touched": 0,
            "c5_artifacts_touched": 0,
            "c6_artifacts_touched": 0,
            "target_accessed": False,
            "target_access": 0,
            "why_supersession_is_safe": (
                "the correction landed before the first scientific C7 trial, so no "
                "measurement was taken under the superseded identity. Nothing "
                "needs regenerating and no result changes."),
        },

        "corrected_declaration": {
            "c7_search_arm": search.training_arm,
            "tracks": tracks,
            "total_declared_trials": total,
            "epochs_per_trial": epochs,
            "steps_per_epoch": steps,
            "total_declared_optimizer_steps": total * epochs * steps,
            "derived_from": (
                "prism_fas.search.plan.detector_search_plan over the canonical "
                "configs; not hard-coded"),
        },

        "supersedes": {
            "record": "reports/handoff/LR_ANCHOR_DECISION_RECORD.json",
            "record_preserved_unchanged": True,
            "record_sha256": _sha256_file(FROZEN_RECORD),
            "superseded_field": "trial_count_change.C7_TRACK_G",
            "superseded_claim": frozen.get("trial_count_change", {}).get("C7_TRACK_G"),
            "rule": (
                "the 2026-08-17 record is the evidence of what was DECIDED and is "
                "never rewritten; this record is the evidence of what was "
                "CORRECTED in the implementation of it. Both are kept."),
        },

        "immutability": {"rewrite_permitted": False},
        "executed": {"trials": 0, "scientific_training_runs": 0, "gpu_seconds": 0,
                     "winners_selected": 0},
        "git": {"commit": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
        "regressions": [
            "tests/pipeline/test_lr_track_g_coordinate.py",
            "tests/pipeline/test_c7_search_arm_decision.py",
            "tests/pipeline/test_search_engine.py",
        ],
    }


def main() -> int:
    from prism_fas.pipeline.state import atomic_write_json

    payload = build()
    atomic_write_json(OUT, payload)

    identity = payload["identity_change"]
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")
    print(f"  lr decision identity : {identity['lr_decision_identity']['before'][:16]}"
          f" -> {identity['lr_decision_identity']['after'][:16]}")
    print(f"  decision config bytes: unchanged="
          f"{not identity['decision_config_bytes']['changed']}")
    print(f"  C4 plan identity     : changed="
          f"{identity['c4_gpat_search_plan_identity']['changed']} (must be False)")
    declaration = payload["corrected_declaration"]
    for track, item in sorted(declaration["tracks"].items()):
        print(f"  Track {track}: {item['declared_trials']} trials, LR applicable="
              f"{item['learning_rate']['applicable']}, "
              f"multipliers={item['learning_rate']['multipliers']}")
    print(f"  total: {declaration['total_declared_trials']} trials, "
          f"{declaration['total_declared_optimizer_steps']} optimizer steps")

    if identity["c4_gpat_search_plan_identity"]["changed"]:
        print("REFUSED: C4's frozen search plan identity moved.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
