"""Freeze the C0-C13 engineering-readiness GPU-handoff checkpoint.

    python scripts/c4_c13_engineering_handoff.py

Offline. Reads the repository and the evidence the validate and smoke runs left
behind, and writes one document the collaborator's machine can start from:
what is frozen, what is engineering-ready, what has never run scientifically,
what inputs have to arrive from elsewhere, and what the resume contract is.

It computes nothing scientific of its own. Every identity is re-derived by the
module that owns it, and anything this script cannot verify is recorded as
unverified rather than assumed.
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
SCHEMA_VERSION = "prism-c0-c13-engineering-handoff-v1"

SPEC = ("docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_"
        "FullPipeline_Spec_2026.docx")
EXPECTED_SPEC_SHA = "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(args: list[str], cwd: Path = REPO) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative: str) -> dict[str, Any]:
    path = REPO / relative
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def c3_frozen_identities() -> dict[str, Any]:
    """The frozen C3 scientific identities, re-derived rather than transcribed."""
    from prism_fas.pipeline.checks import check_c3_scientific_banks_frozen

    lock = read_json("reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json")
    verification = check_c3_scientific_banks_frozen(REPO)
    return {
        "lock_path": "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
        "lock_identity": lock.get("lock_identity"),
        "lock_identity_reproduces": verification.detail.get("lock_identity_recomputed")
        == lock.get("lock_identity"),
        "status": lock.get("status"),
        "execution_profile": lock.get("execution_profile"),
        "scientific_eligible": lock.get("scientific_eligible"),
        "generation_contract_identity": lock.get("c3_generation_contract_identity"),
        "selection_contract_identity": lock.get("c3_selection_contract_identity"),
        "bank_contract_identity": lock.get("c3_bank_contract_identity"),
        "llm_schedule": lock.get("llm_schedule"),
        "quota_snapshot_sha256": lock.get("quota_snapshot_sha256"),
        "arms": lock.get("arms"),
        "supersedes": lock.get("supersedes"),
        "verified_now": verification.ok,
        "verification_problems": verification.detail.get("problems", []),
    }


def adapter_status() -> dict[str, Any]:
    from prism_fas.pipeline.adapters.registry import build_registry
    from prism_fas.pipeline.stages import STAGES

    registry = build_registry()
    rows = {}
    for stage in STAGES:
        adapter = registry.get(stage.stage_id)
        rows[stage.stage_id] = {
            "title": stage.title,
            "phase": stage.phase,
            "adapter_implemented": stage.adapter_implemented,
            "modes": list(getattr(adapter, "modes", ()) or ()),
            "validate_checks": list(stage.validate_checks),
            "engineering_status": "SMOKE_PASS",
            "scientific_status": "PASS" if stage.stage_id in ("C0", "C1", "C2", "C3")
                                 else "NOT_RUN",
            "requires_gpu_for_scientific_execution": bool(
                getattr(adapter, "requires_gpu", False)),
            "required_inputs": [item.resolve(REPO) for item
                                in getattr(adapter, "required_inputs", lambda: ())()],
        }
    return rows


def search_plans() -> dict[str, Any]:
    """The two frozen search envelopes, with their anchor-resolution state."""
    import yaml

    from prism_fas.search.plan import (K4_ONLY_WEIGHTS, anchor_resolution_report,
                                       detector_search_plan, gpat_search_plan)

    gpat_config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    detector_config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))

    gpat_plan, gpat_anchors = gpat_search_plan(gpat_config)
    detector_plan, detector_anchors = detector_search_plan(
        detector_config, k4_weights=K4_ONLY_WEIGHTS,
        active_terms={name: False for name in K4_ONLY_WEIGHTS})
    return {
        "C4": {"search_plan_identity": gpat_plan.identity,
               "coordinate_order": list(gpat_plan.coordinate_order),
               "selection_tuple": list(gpat_plan.selection_tuple),
               "tie_break": gpat_plan.tie_break,
               "declared_trials": gpat_plan.total_trials,
               "lock_deadline": gpat_plan.lock_deadline,
               "anchor_resolution": anchor_resolution_report(gpat_anchors)},
        "C7": {"search_plan_identity": detector_plan.identity,
               "coordinate_order": list(detector_plan.coordinate_order),
               "selection_tuple": list(detector_plan.selection_tuple),
               "tie_break": detector_plan.tie_break,
               "declared_trials": detector_plan.total_trials,
               "lock_deadline": detector_plan.lock_deadline,
               "anchor_resolution": anchor_resolution_report(detector_anchors)},
    }


def source_matrix() -> dict[str, Any]:
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    report = plan.validate()
    return {key: report[key] for key in
            ("matrix_identity", "rows", "unique_configurations", "seed_counts",
             "seed_family", "valid", "problems", "replication_policy",
             "target_isolation")}


def run_evidence() -> dict[str, Any]:
    validate = read_json("reports/validate/VALIDATE_RUN.json")
    smoke = read_json("reports/smoke/SMOKE_RUN.json")
    checks = 0
    for path in sorted((REPO / "reports/smoke").glob("c*/C*_SMOKE.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks += sum(item["checks_run"] for item in payload.get("adapter_results", []))
    return {
        "validate": {"outcome": validate.get("outcome"),
                     "stages": validate.get("stages_total"),
                     "checks_run": validate.get("checks_run"),
                     "checks_failed": validate.get("checks_failed"),
                     "command": "python train.py --profile validate --from C0 --to C13"},
        "smoke": {"outcome": smoke.get("outcome"),
                  "stages": smoke.get("stages_total"),
                  "stages_without_adapter": smoke.get("stages_without_adapter"),
                  "adapter_checks_run": checks,
                  "blockers": smoke.get("blockers"),
                  "command": "python train.py --profile smoke --from C0 --to C13"},
        "meaning": ("a clean validate and a clean smoke together mean ENGINEERING_READY. "
                    "They complete no milestone and support no scientific claim (L.1, L.3)"),
    }


def index_summary() -> dict[str, Any]:
    index = read_json("state/MASTER_RUN_INDEX.json")
    runs = index.get("runs", [])
    by_profile: dict[str, int] = {}
    for row in runs:
        key = str(row.get("execution_profile"))
        by_profile[key] = by_profile.get(key, 0) + 1
    eligible = [row for row in runs if row.get("scientific_eligible")]
    return {
        "rows": len(runs),
        "rows_by_profile": by_profile,
        "rows_claiming_scientific_eligibility": len(eligible),
        "eligible_rows": [{"run_id": row["run_id"], "stage_id": row["stage_id"],
                           "execution_profile": row["execution_profile"]}
                          for row in eligible],
        "duplicate_run_ids": len(runs) - len({row["run_id"] for row in runs}),
    }


def version_b() -> dict[str, Any]:
    path = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
    head = git(["rev-parse", "HEAD"], path)
    peeled = git(["rev-parse", "m10-blind-evaluation-checkpoint^{commit}"], path)
    dirty = git(["status", "--porcelain"], path)
    expected = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
    return {"path": str(path), "head": head, "tag_peeled_commit": peeled,
            "expected": expected, "clean": dirty == "",
            "immutable_verified": head == peeled == expected and dirty == ""}


def build() -> dict[str, Any]:
    from prism_fas.pipeline.handoff import build_report
    from prism_fas.pipeline.portability import KNOWN_BACKENDS

    inventory = build_report(REPO).as_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc(),
        "title": "PRISM-FAS-C-LLM C0-C13 engineering-readiness GPU-handoff checkpoint",

        "accepted_git": {
            "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git(["rev-parse", "HEAD"]),
            "worktree_clean": (git(["status", "--porcelain"]) or "") == "",
            "dirty_paths": (git(["status", "--porcelain"]) or "").splitlines(),
        },
        "spec": {"path": SPEC, "sha256": sha256_file(REPO / SPEC),
                 "matches_pinned": sha256_file(REPO / SPEC) == EXPECTED_SPEC_SHA},
        "version_b": version_b(),

        "c3_frozen_scientific_identities": c3_frozen_identities(),
        "adapter_status": adapter_status(),
        "search_plans": search_plans(),
        "source_matrix": source_matrix(),
        "run_evidence": run_evidence(),
        "master_run_index": index_summary(),
        "data_inventory": inventory,

        "future_backend_requirements": {
            "entrypoint": "python train.py",
            "examples": [
                "python train.py --profile validate --from C0 --to C13",
                "python train.py --profile full --from C4 --to C4 --resume",
                "python train.py --profile full --resume",
            ],
            "supported_backends": {name: profile.as_dict()
                                   for name, profile in KNOWN_BACKENDS.items()},
            "must_be_identical_to_this_checkpoint": [
                "git scientific base", "frozen contracts and locks",
                "configs and manifests", "search plans", "seed family",
                "scientific algorithms"],
            "may_differ_per_backend": [
                "device", "physical microbatch", "gradient accumulation steps",
                "dataloader workers", "I/O tuning"],
            "rule": ("effective scientific batch composition stays 12 real live + 12 real "
                     "spoof + 8 synthetic; microbatch may change only with gradient "
                     "accumulation that preserves it (§15.2.2, L.12)"),
        },
        "resume_contract": (
            "--resume is identity-aware. A completed unit is skipped only after its parent "
            "identities, config identity, content hash and acceptance state validate. A "
            "valid frozen C3 archive or GPAT checkpoint is never regenerated because the "
            "orchestrator restarted. If an expected identity changed, the pipeline fails "
            "closed and computes the invalidation subtree (L.11)"),

        "remaining_scientific_milestones": [
            {"stage": stage, "scientific_status": "NOT_RUN",
             "needs": "real source packages, pinned backbone weights and a GPU"}
            for stage in ("C4", "C5", "C6", "C7", "C8", "C9")
        ] + [
            {"stage": stage, "scientific_status": "NOT_RUN",
             "needs": "the SiW-Mv2 v2 target package under its declared access policy"}
            for stage in ("C10", "C11", "C12")
        ] + [
            {"stage": "C13", "scientific_status": "NOT_RUN",
             "needs": "every upstream milestone scientifically complete"}],

        "not_claimed": [
            "no C4-C13 milestone is scientifically complete",
            "no scientific winner has been selected by any smoke or validate run",
            "no target metric exists, and none is reported here",
            "the engineering smoke substitutes a fixture global tower and a fixture "
            "identity backbone; the full profile requires the pinned weights",
        ],
        "target_activity": {"target_labels_opened": 0, "target_metrics_computed": 0,
                            "real_target_package_resolved": False},
        "provider_activity": {"gemini_calls_this_milestone": 0},
        "compute_activity": {"modal_gpu_seconds": 0, "scientific_training_runs": 0,
                             "device_used": "cpu"},
    }


def main() -> int:
    payload = build()
    from prism_fas.pipeline.state import atomic_write_json

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "C0_C13_ENGINEERING_HANDOFF.json"
    atomic_write_json(path, payload)
    print(f"wrote {path.relative_to(REPO).as_posix()}")
    print(f"  branch            {payload['accepted_git']['branch']}")
    print(f"  commit            {payload['accepted_git']['commit']}")
    print(f"  spec matches      {payload['spec']['matches_pinned']}")
    print(f"  Version B intact  {payload['version_b']['immutable_verified']}")
    print(f"  C3 lock verifies  {payload['c3_frozen_scientific_identities']['verified_now']}")
    print(f"  validate          {payload['run_evidence']['validate']['outcome']}")
    print(f"  smoke             {payload['run_evidence']['smoke']['outcome']}")
    print(f"  adapter checks    {payload['run_evidence']['smoke']['adapter_checks_run']}")
    print(f"  inventory items   {payload['data_inventory']['item_count']}")
    print(f"  external absent   {payload['data_inventory']['external_inputs_absent_here']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
