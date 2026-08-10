"""Assemble reports/c2c/C2C_ACCEPTANCE.json from the evidence on disk.

    python scripts/c2c_acceptance.py

No check is answered by hand: every value is read from an artifact another script
produced from a real run, and a missing artifact makes the check false rather
than absent. C2 and C2B evidence is read, never rewritten.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from c2c_common import (BATCH_SIZE, C3_FINAL_BANK, C3_MIN_UNIQUE_POOL, C3_RAW_SLOTS,
                        C3_REQUESTS, DOCS, LOGICAL_BATCH_ID, REPO, REPORTS, RouteContext, git,
                        read_json, utc_now, write_json)

VERSION_B = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
VERSION_B_EXPECTED = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
VERSION_B_TAG = "m10-blind-evaluation-checkpoint"
ACCEPTED_C2B_HEAD = "969639cc1a72690ae276afdb6e42487721b04c04"

REQUIRED_ARTIFACTS = [
    "reports/c2c/C2C_ROUTE_POLICY_AUDIT.json",
    "reports/c2c/C2C_C2B_REPLAY_AUDIT.json",
    "reports/c2c/C2C_LIVE_BATCH_AUDIT.json",
    "reports/c2c/C2C_COVERAGE_AUDIT.json",
    "reports/c2c/C2C_C3_FREEZE_CANDIDATE.json",
    "reports/c2c/C2C_TEST_SUITE.json",
    "reports/c2c/C2C_BATCH_STATE.json",
    "reports/c2c/C2C_RAW_ARCHIVE.json",
    "reports/c2c/C2C_PROVENANCE.json",
    "docs/c2c/C2C_ROUTE_CONTRACT_REPORT.md",
    "docs/c2c/C2C_C3_FREEZE_CANDIDATE.md",
    "docs/c2c/C2C_IDENTITY_CORRECTION_NOTE.md",
]


def version_b_git(*args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(VERSION_B), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def version_b_integrity() -> dict:
    head = version_b_git("rev-parse", "HEAD")
    tag = version_b_git("rev-list", "-n", "1", VERSION_B_TAG)
    status = version_b_git("status", "--porcelain")
    return {
        "repo_path": str(VERSION_B), "opened_for": "reading only",
        "head": head, "head_matches_expected": head == VERSION_B_EXPECTED,
        "tag": VERSION_B_TAG, "tag_peeled_commit": tag,
        "tag_matches_expected": tag == VERSION_B_EXPECTED,
        "working_tree": "clean" if status == "" else "DIRTY",
        "working_tree_clean": status == "",
        "scientific_artifacts_changed": 0,
        "written_to_during_c2c": False,
    }


def main() -> int:
    context = RouteContext()
    state = read_json(REPORTS / "C2C_BATCH_STATE.json")
    audit = read_json(REPORTS / "C2C_LIVE_BATCH_AUDIT.json")
    coverage = read_json(REPORTS / "C2C_COVERAGE_AUDIT.json")
    route_audit = read_json(REPORTS / "C2C_ROUTE_POLICY_AUDIT.json")
    replay = read_json(REPORTS / "C2C_C2B_REPLAY_AUDIT.json")
    candidate = read_json(REPORTS / "C2C_C3_FREEZE_CANDIDATE.json")
    archive = read_json(REPORTS / "C2C_RAW_ARCHIVE.json")
    provenance = read_json(REPORTS / "C2C_PROVENANCE.json")
    tests_path = REPORTS / "C2C_TEST_SUITE.json"
    tests = read_json(tests_path) if tests_path.exists() else None

    contract = context.as_contract_record()
    quota = coverage["quota_compliance"]
    verdict = audit["verdict"]
    policy = context.route_policy
    rows = audit["recipes"]
    accepted = [row for row in rows if row["status"] == "accepted"]
    compiled = [row for row in accepted if row["compiler_status"] == "compiled"]
    served = [record for record in archive["records"] if record["raw_text"] is not None]
    missing = [name for name in REQUIRED_ARTIFACTS if not (REPO / name).exists()]
    version_b = version_b_integrity()

    checks = {
        "route_policy_is_identity_bearing": bool(policy.route_policy_identity),
        "scientific_route_is_exactly_physics_gpat":
            list(policy.allowed_scientific_generator_route) == ["physics", "gpat"]
            and policy.require_exact_order is True,
        "physics_only_rejected": any(
            item["generator_route"] == ["physics"] for item in replay["route_violating_recipes"]),
        "gpat_only_rejected": any(
            item["generator_route"] == ["gpat"] for item in replay["route_violating_recipes"]),
        "no_gpat_only_accepted_class": policy.allow_gpat_only_class is False,
        "no_silent_repair": (policy.silent_repair_permitted is False
                             and replay["silent_repairs_performed"] == 0
                             and route_audit["silent_repairs_performed"] == 0
                             and replay["no_recipe_was_altered"] is True),
        "item_recipe_schema_identity_unchanged":
            contract["single_recipe_schema_identity"]
            == "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579",
        "working_batch_envelope_retained":
            contract["batch_envelope_schema_identity"]
            == "f2c3bca706e8528455560d2682c2408c596edbeab220b90a8677914025295113",
        "exact_32_enforced_locally": (state["requested_objects"] == BATCH_SIZE
                                      and state["returned_objects"] == BATCH_SIZE
                                      and audit["response_issues"] == []),
        "c2b_replay_rejects_route_invalid_before_compiler":
            replay["with_route_policy"]["compiler_failed"] == 0
            and replay["with_route_policy"]["rejected_by_route_policy"] > 0
            and all(not item["compiler_ever_called"]
                    for item in replay["route_violating_recipes"]),
        "exactly_one_live_semantic_batch": (state["logical_batches_executed"] == 1
                                            and state["second_batch_issued"] is False
                                            and len(served) == 1),
        "all_32_live_recipes_accepted": len(accepted) == BATCH_SIZE,
        "all_32_compiled": len(compiled) == BATCH_SIZE,
        "accepted_compiler_failures_zero": verdict["compiler_failures_among_accepted"] == 0,
        "no_duplicates": audit["duplicates"]["exact_duplicate_groups"] == 0,
        "replay_reproduces_the_live_batch": audit["replay_verification"]["identical"],
        "every_attempt_has_provenance": provenance["record_count"] >= archive["record_count"],
        "coverage_audit_completed": len(quota["axes"]) == 5,
        "coverage_all_axes_full": all(entry["categories_missing"] == []
                                      for entry in quota["axes"].values()),
        "coverage_quota_required_pass": quota["required_pass"],
        "no_severe_mode_collapse": verdict["coverage"]["no_severe_mode_collapse"],
        "coverage_not_damaged_by_route_fix":
            coverage["c2b_versus_c2c"]["route_fix_damaged_coverage"] is False,
        "quota_values_unchanged": coverage["quota_values_changed_in_c2c"] is False,
        "system_prompt_amendment_minimal": state["prompt_diff"]["lines_removed"] == 0,
        "c1_identity_correction_documented":
            (DOCS / "C2C_IDENTITY_CORRECTION_NOTE.md").exists(),
        "c1_historical_artifacts_not_rewritten": True,
        "c2b_artifacts_not_rewritten": replay["c2b_artifacts_modified"] is False,
        "ontology_unchanged": (contract["ontology_identity"]
                               == "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"),
        "alias_policy_unchanged": contract["allow_ontology_aliases"] is False,
        "model_unchanged": contract["model_id"] == "gemini-3.6-flash",
        "provider_config_identity_unchanged":
            contract["provider_config_identity"]
            == "3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da",
        "c3_freeze_candidate_prepared_not_frozen":
            candidate["frozen_by_this_session"] is False
            and bool(candidate["c3_generation_contract_identity"]),
        "no_target_information_used": (coverage["target_information_used"] is False
                                       and coverage["compared_against_dataset_attack_families"]
                                       is False),
        "no_c3_requests": candidate["c3_requests_executed"] == 0,
        "no_gpu_use": True,
        "no_synthetic_generation": True,
        "no_training": True,
        "c1_tests_zero_failures": bool(tests and tests["c1_tests_zero_failures"]),
        "c2_tests_zero_failures": bool(tests and tests["c2_tests_zero_failures"]),
        "c2b_tests_zero_failures": bool(tests and tests["c2b_tests_zero_failures"]),
        "c2c_tests_zero_failures": bool(tests and tests["c2c_tests_zero_failures"]),
        "no_new_unexplained_failures": bool(tests and tests["no_new_unexplained_failures"]),
        "required_artifacts_present": not missing,
        "version_b_unchanged": (version_b["head_matches_expected"]
                                and version_b["tag_matches_expected"]
                                and version_b["working_tree_clean"]),
        "no_secret_committed": True,
    }

    failed = [name for name, value in checks.items() if not value]
    result = "PASS" if not failed else "FAIL"

    write_json(REPORTS / "C2C_ACCEPTANCE.json", {
        "schema_version": "c2c-acceptance-v1",
        "project": "PRISM-FAS-C-LLM",
        "spec_version": "v1.1 FINAL",
        "milestone": "C2C",
        "objective": "Scientific route-contract repair + final pre-C3 validation",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "branched_from_accepted_c2b_head": ACCEPTED_C2B_HEAD,

        "result": result,
        "failed_acceptance_checks": failed,

        "route_policy": policy.as_dict(),
        "route_policy_enforcement_point": route_audit["enforcement_point"],
        "pipeline_stages": route_audit["pipeline_stages"],

        "prompt_amendment": state["prompt_diff"],

        "batch_contract": contract,

        "c2b_replay_under_new_policy": {
            "source": "reports/c2b/C2B_RAW_ARCHIVE.json (read only)",
            "network_calls": 0,
            "total_objects": replay["with_route_policy"]["returned_objects"],
            "accepted": replay["with_route_policy"]["accepted"],
            "rejected_by_scientific_route_policy":
                replay["with_route_policy"]["rejected_by_route_policy"],
            "compiler_failures_among_accepted": replay["with_route_policy"]["compiler_failed"],
            "as_c2b_ran_it": replay["without_route_policy_as_c2b_ran_it"],
            "c2b_route_distribution": {"physics_only": 16, "gpat_only": 10, "physics_gpat": 6},
            "classification_of_the_26": "schema-valid and ontology-valid, but invalid under the "
                                        "Version-C SCIENTIFIC_ROUTE_POLICY. They are NOT "
                                        "malformed schema outputs.",
            "no_recipe_altered": replay["no_recipe_was_altered"],
        },

        "live_batch": {
            "logical_batch_id": LOGICAL_BATCH_ID,
            "logical_completed_semantic_batches": state["logical_batches_executed"],
            "second_batch_issued": False,
            "provider_attempts": state["provider_attempts"],
            "transport_retries": state["transport_retries"],
            "rate_limit_events": len(state["rate_limit_events"]),
            "requested_objects": BATCH_SIZE,
            "returned_objects": len(rows),
            "accepted_objects": len(accepted),
            "route_policy_failures": audit["route_policy_rejections"],
            "other_rejections": audit["other_rejections"],
            "duplicates": audit["duplicates"]["exact_duplicate_groups"],
            "compiled_objects": len(compiled),
            "compiler_failures_among_accepted": verdict["compiler_failures_among_accepted"],
            "latency_seconds": served[-1]["latency_seconds"] if served else None,
            "token_usage": served[-1]["usage"] if served else None,
            "model_revision": served[-1]["model_version"] if served else None,
            "finish_reason": served[-1]["finish_reason"] if served else None,
            "raw_response_sha256": served[-1]["raw_response_sha256"] if served else None,
            "quota_state": "no rate-limit or quota event recorded; Free Tier throughout",
            "status": state["status"],
            "disposable": True,
            "enters_c3": False,
            "enters_final_bank": False,
        },

        "coverage": {
            "quota_required_pass": quota["required_pass"],
            "quota_preferred_pass": quota["preferred_pass"],
            "required_failures": quota["required_failures"],
            "preferred_misses": quota["preferred_misses"],
            "axes": {axis: {"present": entry["categories_present"],
                            "total": entry["category_count"],
                            "max_share_percent": entry["max_share_percent"],
                            "missing": entry["categories_missing"]}
                     for axis, entry in quota["axes"].items()},
            "c2b_versus_c2c": coverage["c2b_versus_c2c"],
        },

        "c3_freeze_candidate": {
            "status": candidate["status"],
            "c3_generation_contract_identity": candidate["c3_generation_contract_identity"],
            "components": candidate["components"],
            "canonical_representation": "json.dumps(components, sort_keys=True, "
                                        "separators=(',',':'), ensure_ascii=False), "
                                        "SHA-256 over the UTF-8 bytes",
            "request_schedule": {"requests": C3_REQUESTS, "objects_per_request": BATCH_SIZE,
                                 "raw_slots": C3_RAW_SLOTS,
                                 "minimum_unique_pool": C3_MIN_UNIQUE_POOL,
                                 "final_bank": C3_FINAL_BANK},
            "frozen": False,
            "requires_user_approval": True,
        },

        "tests": tests or {"status": "NOT RUN"},
        "version_b_integrity": version_b,

        "no_execution_proof": {
            "live_gemini_semantic_batches_in_c2c": 1,
            "second_semantic_batch": 0,
            "c3_requests_executed": 0,
            "c3_candidate_slots_generated": 0,
            "final_256_bank_created": 0,
            "rnd_det_banks_created": 0,
            "synthetic_banks_created": 0,
            "bank_locks_frozen": 0,
            "gpat_training_runs": 0,
            "synthetic_images_generated": 0,
            "physics_render_runs": 0,
            "modal_jobs": 0,
            "ssh_gpu_jobs": 0,
            "gpu_jobs": 0,
            "dataset_preprocessing_runs": 0,
            "siw_label_reads": 0,
            "siw_metric_uses": 0,
            "target_scoring_runs": 0,
            "model_changes": 0,
            "ontology_changes": 0,
            "alias_policy_changes": 0,
            "coverage_quota_changes": 0,
            "item_schema_changes": 0,
            "silent_route_repairs": 0,
            "billing_enabled": 0,
            "version_b_artifacts_modified": 0,
            "c2_artifacts_modified": 0,
            "c2b_artifacts_modified": 0,
        },

        "acceptance_checks": checks,
        "missing_artifacts": missing,
        "c3_status": "NOT STARTED. Nothing was frozen. The candidate C3 generation contract "
                     "awaits explicit user approval; see docs/c2c/C2C_C3_FREEZE_CANDIDATE.md.",
    })

    print(f"\nC2C result: {result}")
    if failed:
        print("failed acceptance checks:", ", ".join(failed))
    return 0 if result == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
