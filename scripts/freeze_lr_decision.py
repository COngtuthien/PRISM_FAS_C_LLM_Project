"""Freeze the approved learning-rate decision and the plans it produces.

    python scripts/freeze_lr_decision.py

Offline and read-only with respect to science: it builds the two search plans in
memory, records their identities, and writes one immutable record. It executes no
trial and selects no winner.

The awaiting-approval dossier is preserved unchanged beside it. The pair is the
audit trail — what was asked, and what was decided.
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

OUT = REPO / "reports" / "handoff"
SCHEMA_VERSION = "prism-lr-anchor-decision-record-v1"

#: The plan identities the search plans had while the decision was open. Recorded
#: so the supersession is legible: these are what the plans hashed to when
#: `learning_rate` was AMBIGUOUS and contributed no trials.
PRE_DECISION_IDENTITIES = {
    "c4_gpat_coordinate_v1":
        "ab77e964d9c035cf2c3bed209ffac307aebd85c6735879bc3fa3c5efce20d0ec",
    "c7_detector_coordinate_v1":
        "62d0022507e732ba89618845fab2c63fec2b7b07f6817b2d541a4f500f459d7b",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build() -> dict[str, Any]:
    import yaml

    from prism_fas.search.lr_decision import load_decision
    from prism_fas.search.plan import (K4_ONLY_WEIGHTS, anchor_resolution_report,
                                       detector_search_plan, gpat_search_plan)

    record = load_decision(REPO)
    if not record.approved:
        raise SystemExit("the decision record is not APPROVED; refusing to freeze")

    gpat_config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    detector_config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))

    c4_plan, c4_anchors = gpat_search_plan(
        gpat_config, lr_decision=record.for_component("C4"))
    r_plan, r_anchors = detector_search_plan(
        detector_config, k4_weights=K4_ONLY_WEIGHTS,
        active_terms={name: False for name in K4_ONLY_WEIGHTS},
        lr_decision=record.for_component("C7_TRACK_R"))
    g_plan, _g_anchors = detector_search_plan(
        detector_config,
        active_terms={"lambda_local": False, "lambda_MIL": False, "lambda_P": False},
        lr_decision=record.for_component("C7_TRACK_G"))

    def plan_block(plan: Any, component: str) -> dict[str, Any]:
        decision = record.for_component(component)
        return {
            "plan_id": plan.plan_id,
            "search_plan_identity": plan.identity,
            "coordinate_order": list(plan.coordinate_order),
            "active_coordinates": [item.name for item in plan.active_coordinates],
            "total_trials": plan.total_trials,
            "selection_tuple": list(plan.selection_tuple),
            "tie_break": plan.tie_break,
            "one_pass": plan.one_pass,
            "lock_deadline": plan.lock_deadline,
            "lr_decision": decision.as_dict(),
            "lr_ratio_preserved_at_every_multiplier": all(
                decision.ratio_preserved(value) for value in decision.candidates) or
                not decision.searches_a_multiplier,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc(),
        "status": "FROZEN_APPROVED_LR_DECISION",
        "decision_status": "APPROVED",
        "decision_identity": record.identity,
        "decision_config": record.config_path,
        "decision_config_sha256": record.config_sha256,
        "supersedes": {
            "dossier": record.dossier,
            "dossier_identity": record.dossier_identity,
            "dossier_status_before": "AWAITING_USER_APPROVAL",
            "dossier_preserved_unchanged": True,
            "rule": "the dossier is the question and this record is the answer; both are "
                    "kept so the decision remains auditable",
        },
        "pre_decision_search_plan_identities": PRE_DECISION_IDENTITIES,
        "frozen_search_plans": {
            "C4": plan_block(c4_plan, "C4"),
            "C7_TRACK_R": plan_block(r_plan, "C7_TRACK_R"),
            "C7_TRACK_G": plan_block(g_plan, "C7_TRACK_G"),
        },
        "trial_count_change": {
            "C4": {"before": 9, "after": c4_plan.total_trials},
            "C7_TRACK_R": {"before": 21, "after": r_plan.total_trials},
            "C7_TRACK_G": {"before": 12, "after": g_plan.total_trials,
                           "note": "unchanged; Track G's LR coordinate is inapplicable "
                                   "because backbone_lr controls no Track-G parameter"},
        },
        "anchor_resolution_after_decision": {
            "C4": anchor_resolution_report(c4_anchors),
            "C7_TRACK_R": anchor_resolution_report(r_anchors),
        },
        "git": {"branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "commit": git(["rev-parse", "HEAD"])},
        "executed": {"trials": 0, "winners_selected": 0, "gpu_seconds": 0,
                     "scientific_training_runs": 0},
        "immutability": {"rewrite_permitted": False},
        "meaning": "the C4 and C7 bounded envelopes are now scientifically executable. "
                   "This record freezes WHAT they will search; it runs nothing.",
    }


def main() -> int:
    from prism_fas.pipeline.state import atomic_write_json

    payload = build()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "LR_ANCHOR_DECISION_RECORD.json"
    atomic_write_json(path, payload)

    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    print(f"wrote {path.relative_to(REPO).as_posix()}")
    print(f"  record identity     {identity}")
    print(f"  decision identity   {payload['decision_identity']}")
    for name, block in payload["frozen_search_plans"].items():
        print(f"  {name:<11} plan {block['search_plan_identity'][:16]}  "
              f"trials={block['total_trials']:<3} "
              f"lr={block['lr_decision']['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
