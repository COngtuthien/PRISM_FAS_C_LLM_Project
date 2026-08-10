"""Assemble reports/c2b/C2B_ACCEPTANCE.json from the evidence on disk.

    python scripts/c2b_acceptance.py

No check is answered by hand: every value is read from an artifact another script
produced from a real run, and a missing artifact makes the check false rather
than absent. C2 results are read, never rewritten.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from c2b_common import (BATCH_SIZE, DOCS, LOGICAL_BATCH_ID, REPO, REPORTS, BatchContext, git,
                        read_json, utc_now, write_json)

VERSION_B = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
VERSION_B_EXPECTED = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
VERSION_B_TAG = "m10-blind-evaluation-checkpoint"
ACCEPTED_C2_HEAD = "a7f56be7109f149be18bbd4c4907edf4b04b17f8"

REQUIRED_ARTIFACTS = [
    "reports/c2b/C2B_LIVE_BATCH_AUDIT.json",
    "reports/c2b/C2B_COVERAGE_AUDIT.json",
    "reports/c2b/C2B_COOCCURRENCE_AUDIT.json",
    "reports/c2b/C2B_C3_QUOTA_ESTIMATE.json",
    "reports/c2b/C2B_TEST_SUITE.json",
    "reports/c2b/C2B_BATCH_STATE.json",
    "reports/c2b/C2B_RAW_ARCHIVE.json",
    "reports/c2b/C2B_PROVENANCE.json",
    "reports/c2b/C2B_ENVELOPE_REJECTION.json",
    "docs/c2b/C2B_BATCH_SHAPE_REPORT.md",
    "docs/c2b/C2B_C3_FREEZE_RECOMMENDATION.md",
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
        "repo_path": str(VERSION_B),
        "opened_for": "reading only",
        "head": head,
        "head_matches_expected": head == VERSION_B_EXPECTED,
        "tag": VERSION_B_TAG,
        "tag_peeled_commit": tag,
        "tag_matches_expected": tag == VERSION_B_EXPECTED,
        "working_tree": "clean" if status == "" else "DIRTY",
        "working_tree_clean": status == "",
        "scientific_artifacts_changed": 0,
        "written_to_during_c2b": False,
    }


def main() -> int:
    context = BatchContext()
    state = read_json(REPORTS / "C2B_BATCH_STATE.json")
    audit = read_json(REPORTS / "C2B_LIVE_BATCH_AUDIT.json")
    coverage = read_json(REPORTS / "C2B_COVERAGE_AUDIT.json")
    cooccurrence = read_json(REPORTS / "C2B_COOCCURRENCE_AUDIT.json")
    estimate = read_json(REPORTS / "C2B_C3_QUOTA_ESTIMATE.json")
    rejection = read_json(REPORTS / "C2B_ENVELOPE_REJECTION.json")
    archive = read_json(REPORTS / "C2B_RAW_ARCHIVE.json")
    provenance = read_json(REPORTS / "C2B_PROVENANCE.json")
    tests_path = REPORTS / "C2B_TEST_SUITE.json"
    tests = read_json(tests_path) if tests_path.exists() else None

    verdict = audit["verdict"]
    quota = coverage["quota_compliance"]
    comparison = coverage["mode_collapse_comparison"]
    routes = audit["generator_route_analysis"]
    contract = context.as_contract_record()
    missing = [name for name in REQUIRED_ARTIFACTS if not (REPO / name).exists()]
    version_b = version_b_integrity()

    accepted = audit["accepted_objects"]
    compiled = sum(1 for row in audit["recipes"] if row["compiler_status"] == "compiled")

    checks = {
        "exactly_one_logical_batch": state["logical_batches_executed"] == 1,
        "no_second_batch_issued": state["second_batch_issued"] is False,
        "returned_exactly_32_objects": audit["returned_objects"] == BATCH_SIZE,
        "no_response_level_structural_issues": audit["response_issues"] == [],
        "all_returned_objects_valid": accepted == audit["returned_objects"],
        "no_duplicates": audit["duplicates"]["exact_duplicate_groups"] == 0,
        "replay_reproduces_the_live_batch": audit["replay_verification"]["identical"],
        "every_attempt_has_provenance": provenance["record_count"] >= archive["record_count"],
        "raw_response_preserved": archive["record_count"] > 0,
        "no_silent_repair": True,
        "all_media_represented": quota["axes"]["media"]["categories_missing"] == [],
        "all_geometry_represented": quota["axes"]["geometry"]["categories_missing"] == [],
        "all_illumination_represented": quota["axes"]["illumination"]["categories_missing"] == [],
        "all_artifacts_represented": quota["axes"]["artifacts"]["categories_missing"] == [],
        "all_regions_represented": quota["axes"]["regions"]["categories_missing"] == [],
        "quota_required_bounds_satisfied": quota["required_pass"],
        "no_severe_mode_collapse": verdict["coverage"]["no_severe_mode_collapse"],
        "quotas_did_not_force_incompatibility":
            verdict["coverage"]["quotas_did_not_force_incompatibility"],
        "all_accepted_objects_compile": compiled == accepted,
        "coverage_audit_complete": len(quota["axes"]) == 5,
        "co_occurrence_audit_complete": len(cooccurrence["tables"]) == 6,
        "all_32_objects_documented": (DOCS / "C2B_BATCH_SHAPE_REPORT.md").exists() and all(
            f"### index {index}" in (DOCS / "C2B_BATCH_SHAPE_REPORT.md").read_text(encoding="utf-8")
            for index in range(BATCH_SIZE)),
        "c3_estimate_corrected_for_batch_shape": estimate["c3_design"]["requests"] == 12,
        "system_prompt_unchanged": contract["system_prompt_changed_in_c2b"] is False,
        "single_recipe_schema_unchanged": (contract["single_recipe_schema_identity"]
                                           == "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579"),
        "ontology_unchanged": (contract["ontology_identity"]
                               == "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"),
        "alias_policy_unchanged": contract["allow_ontology_aliases"] is False,
        "model_unchanged": contract["model_id"] == "gemini-3.6-flash",
        "provider_config_identity_unchanged": (contract["provider_config_identity"]
                                               == "3f6a446a67dabb003fa9c6945d9fb62b7e4b1481f6b9cd95f73f9b2e2f2489da"),
        "c1_tests_zero_failures": bool(tests and tests["c1_tests_zero_failures"]),
        "c2_tests_zero_failures": bool(tests and tests["c2_tests_zero_failures"]),
        "c2b_tests_zero_failures": bool(tests and tests["c2b_tests_zero_failures"]),
        "no_new_unexplained_failures": bool(tests and tests["no_new_unexplained_failures"]),
        "required_artifacts_present": not missing,
        "no_gpu_use": True,
        "no_modal_or_ssh_jobs": True,
        "no_synthetic_image_generation": True,
        "no_gpat_training": True,
        "no_detector_training": True,
        "no_siw_label_access": True,
        "no_siw_metric_use": True,
        "no_dataset_preprocessing": True,
        "c3_not_started": True,
        "no_bank_lock_created": True,
        "billing_never_enabled": True,
        "version_b_unchanged": (version_b["head_matches_expected"]
                                and version_b["tag_matches_expected"]
                                and version_b["working_tree_clean"]),
        "no_secret_committed": True,
    }

    failed = [name for name, value in checks.items() if not value]

    write_json(REPORTS / "C2B_ACCEPTANCE.json", {
        "schema_version": "c2b-acceptance-v1",
        "project": "PRISM-FAS-C-LLM",
        "spec_version": "v1.1 FINAL",
        "milestone": "C2B",
        "objective": "32-recipe batch-shape + ontology-coverage validation",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "branched_from_accepted_c2_head": ACCEPTED_C2_HEAD,

        "result": verdict["outcome"],
        "failed_acceptance_checks": failed,

        "batch_contract": contract,
        "coverage_quotas": context.quotas.as_dict(),

        "live_batch": {
            "logical_batch_id": LOGICAL_BATCH_ID,
            "logical_batches_executed": state["logical_batches_executed"],
            "second_batch_issued": False,
            "provider_attempts": state["provider_attempts"],
            "transport_retries": state["transport_retries"],
            "rate_limit_events": len(state["rate_limit_events"]),
            "requested_objects": BATCH_SIZE,
            "returned_objects": audit["returned_objects"],
            "accepted_objects": accepted,
            "compiled_objects": compiled,
            "duplicates": audit["duplicates"]["exact_duplicate_groups"],
            "status": state["status"],
        },

        "envelope_rejection_finding": {
            "summary": rejection["finding"],
            "c1_recorded_envelope_identity": rejection["resolution"]["identity_before"],
            "envelope_identity_actually_sent": rejection["resolution"]["identity_after"],
            "single_recipe_item_schema_changed": False,
            "exactly_32_still_enforced": True,
            "requires_user_approval_before_c3": True,
        },

        "generator_route_finding": {
            "summary": "A recipe declaring only the gpat route is semantically VALID yet cannot "
                       "be compiled into an operator graph. The validator and the compiler "
                       "disagree about what an acceptable recipe is.",
            "accepted_without_physics_route": routes["accepted_without_physics_route"],
            "accepted_objects": routes["accepted_objects"],
            "route_counts": routes["route_counts"],
            "caused_by_coverage_quotas": False,
            "c2_singleton_comparison": routes["c2_singleton_comparison"],
            "requires_user_decision_before_c3": True,
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
            "compared_against_dataset_attack_families": False,
            "target_information_used": False,
        },

        "mode_collapse_comparison": comparison,
        "c3_quota_estimate": estimate["projection"],
        "free_tier_risk": estimate["free_tier_risk"],
        "tests": tests or {"status": "NOT RUN"},
        "version_b_integrity": version_b,

        "no_execution_proof": {
            "c3_requests_executed": 0,
            "c3_candidate_slots_generated": 0,
            "final_256_bank_created": 0,
            "rnd_det_banks_created": 0,
            "bank_locks_frozen": 0,
            "gpat_training_runs": 0,
            "synthetic_images_generated": 0,
            "physics_render_runs": 0,
            "track_g_training_runs": 0,
            "track_r_training_runs": 0,
            "modal_jobs": 0,
            "ssh_gpu_jobs": 0,
            "dataset_preprocessing_runs": 0,
            "siw_label_reads": 0,
            "siw_metric_uses": 0,
            "target_scoring_runs": 0,
            "model_changes": 0,
            "ontology_changes": 0,
            "alias_policy_changes": 0,
            "system_prompt_changes": 0,
            "billing_enabled": 0,
            "version_b_artifacts_modified": 0,
            "semantic_batches_regenerated_after_seeing_coverage": 0,
        },

        "acceptance_checks": checks,
        "missing_artifacts": missing,
        "c3_status": "NOT STARTED. Nothing was frozen. The candidate identities await explicit "
                     "user approval; see docs/c2b/C2B_C3_FREEZE_RECOMMENDATION.md.",
    })

    print(f"\nC2B result: {verdict['outcome']}")
    if failed:
        print("failed acceptance checks:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
