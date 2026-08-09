"""Read-only Version-B scientific-state fingerprint for PRISM-FAS-C C0.

Writes a JSON snapshot. Opens Version B for READING ONLY. Never writes into B.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

B = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
OUT = Path(sys.argv[1])

NOT_FOUND = "NOT FOUND"


def sha256_file(p: Path) -> str:
    if not p.is_file():
        return NOT_FOUND
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel: str) -> dict:
    p = B / rel
    if not p.exists():
        return {"path": rel, "status": NOT_FOUND}
    return {"path": rel, "status": "present", "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p)}


def jload(rel: str):
    p = B / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(B), *args], capture_output=True,
                           text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else f"ERROR: {r.stderr.strip()}"
    except Exception as exc:  # pragma: no cover
        return f"ERROR: {exc}"


def dig(obj, *keys, default=NOT_FOUND):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


sml = jload("reports/m10/SOURCE_MATRIX_LOCK.json")
m10acc = jload("reports/m10/M10_ACCEPTANCE.json")
m7lock = jload("assets/recipe_banks/prism_recipe_bank_m7_v1/BANK_LOCK.json")
m8lock = jload("data/processed/prism_synthetic_bank_m8_v3_e84c78cd2a9b/BANK_LOCK.json")
m9acc = jload("reports/m9/M9_ACCEPTANCE.json")
pkg = jload("data/processed/prism_data_v1_m3b/PACKAGE_LOCK.json")
tgtpkg = jload("data/processed/prism_target_eval_v2/PACKAGE_LOCK.json")
tgtlab = jload("data/evaluation_only/prism_target_v2_labels/TARGET_LABEL_LOCK.json")
lockset = jload("reports/m10/TARGET_PREDICTION_LOCKSET.json")
tests = jload("reports/m10/TEST_SUITE.json")

snapshot = {
    "schema_version": "c0-version-b-integrity-v1",
    "purpose": ("Read-only fingerprint of the frozen PRISM-FAS-B scientific state, "
                "captured by PRISM-FAS-C C0. Version B was opened for reading only."),
    "version_b_repo_path": str(B),
    "git": {
        "status_short": git("status", "--short"),
        "status_clean": git("status", "--short") == "",
        "head": git("rev-parse", "HEAD"),
        "main": git("rev-parse", "main"),
        "origin_main": git("rev-parse", "origin/main"),
        "tag_m10_blind_evaluation_checkpoint_peeled":
            git("rev-parse", "m10-blind-evaluation-checkpoint^{}"),
        "remotes": git("remote", "-v"),
        "log_1": git("log", "-1", "--decorate", "--oneline"),
        "tags": git("tag").splitlines(),
        "expected_checkpoint": "7799f7decd35db6987ce4578824e5bd8d9eab4ae",
    },
    "frozen_source_package": {
        "package_id": dig(sml, "frozen_inputs", "source_package_id"),
        "package_identity": dig(sml, "frozen_inputs", "source_package_identity"),
        "package_lock_file": entry("data/processed/prism_data_v1_m3b/PACKAGE_LOCK.json"),
        "package_lock_status": dig(pkg, "status") if pkg else NOT_FOUND,
    },
    "m7_recipe_bank": {
        "bank_id": dig(m7lock, "bank_id"),
        "bank_content_identity_sha256": dig(m7lock, "bank_content_identity_sha256"),
        "recipe_count": dig(m7lock, "recipe_count"),
        "status": dig(m7lock, "status"),
        "ontology_version": dig(m7lock, "ontology_version"),
        "ontology_sha256": dig(m7lock, "ontology_sha256"),
        "prompt_sha256": dig(m7lock, "prompt_sha256"),
        "compiler_version": dig(m7lock, "compiler_version"),
        "conditioning": dig(m7lock, "conditioning"),
        "bank_seed": dig(m7lock, "bank_seed"),
        "generator": dig(m7lock, "generator"),
        "bank_lock_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/BANK_LOCK.json"),
        "recipes_jsonl_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/recipes.jsonl"),
        "generator_json_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/generator.json"),
        "prompt_txt_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/prompt.txt"),
        "ontology_yaml_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/ontology.yaml"),
    },
    "m8_synthetic_bank": {
        "bank_id_from_source_matrix_lock": dig(sml, "frozen_inputs", "m8_bank_id"),
        "bank_identity_from_source_matrix_lock": dig(sml, "frozen_inputs", "m8_bank_identity"),
        "bank_lock_keys": sorted(m8lock.keys()) if isinstance(m8lock, dict) else NOT_FOUND,
        "bank_lock_file": entry(
            "data/processed/prism_synthetic_bank_m8_v3_e84c78cd2a9b/BANK_LOCK.json"),
        "a02_random_operator_recipe_bank_identity":
            dig(sml, "artifact_identities", "a02_random_operator_recipe_bank"),
        "a02_random_operator_synthetic_bank_identity":
            dig(sml, "artifact_identities", "a02_random_operator_synthetic_bank"),
        "a02_conditioning_control_identity":
            dig(sml, "artifact_identities", "a02_conditioning_control"),
    },
    "m9_reference": {
        "milestone": dig(m9acc, "milestone"),
        "target_test_opened": dig(m9acc, "target_test_opened"),
        "identities": dig(m9acc, "identities"),
        "prompt_head_cache_identity": dig(m9acc, "prompt_head", "cache_identity"),
        "acceptance_file": entry("reports/m9/M9_ACCEPTANCE.json"),
        "reference_run_file": entry("reports/m9/reference_run.json"),
        "m9_reference_binding_accepted": dig(sml, "m9_reference_binding", "binding_accepted"),
    },
    "m10_experiment_matrix": {
        "m10_matrix_identity": dig(sml, "m10_matrix_identity"),
        "registry_identity": dig(sml, "registry_identity"),
        "logical_rows": dig(sml, "logical_rows"),
        "executable_rows": dig(sml, "executable_rows"),
        "blocked_rows": dig(sml, "blocked_rows"),
        "failed_rows": dig(sml, "failed_rows"),
        "rows_by_status": dig(sml, "rows_by_status"),
        "acceptance_passed": dig(m10acc, "passed"),
        "acceptance_identity": dig(m10acc, "acceptance_identity"),
        "acceptance_failed_checks": dig(m10acc, "failed_checks"),
        "registry_file": entry("reports/m10/M10_REGISTRY.json"),
        "matrix_plan_file": entry("reports/m10/M10_MATRIX_PLAN.json"),
        "acceptance_file": entry("reports/m10/M10_ACCEPTANCE.json"),
        "statistics_file": entry("reports/m10/statistics.json"),
        "summary_file": entry("reports/m10/summary.json"),
        "reliability_file": entry("reports/m10/RELIABILITY.json"),
        "report_html_file": entry("reports/m10/report.html"),
    },
    "source_matrix_lock": {
        "source_matrix_lock_identity": dig(sml, "source_matrix_lock_identity"),
        "schema_version": dig(sml, "source_matrix_lock_schema_version"),
        "selection_rule": dig(sml, "selection_rule"),
        "selection_used_target": dig(sml, "selection_used_target"),
        "target_labels_opened_at_lock_time": dig(sml, "target_labels_opened"),
        "frozen_inputs": dig(sml, "frozen_inputs"),
        "code_lineage": dig(sml, "code_lineage"),
        "file": entry("reports/m10/SOURCE_MATRIX_LOCK.json"),
        "reverification_file": entry("reports/m10/SOURCE_MATRIX_REVERIFICATION.json"),
    },
    "target_package_identity": {
        "target_feature_package_identity_from_lockset":
            dig(lockset, "target_feature_package_identity"),
        "target_package_lock_file":
            entry("data/processed/prism_target_eval_v2/PACKAGE_LOCK.json"),
        "target_package_lock_status": dig(tgtpkg, "status") if tgtpkg else NOT_FOUND,
        "target_label_lock_file":
            entry("data/evaluation_only/prism_target_v2_labels/TARGET_LABEL_LOCK.json"),
        "target_label_lock": tgtlab if isinstance(tgtlab, dict) else NOT_FOUND,
        "target_prediction_lockset_identity": dig(lockset, "lockset_identity"),
        "target_prediction_lockset_status": dig(lockset, "status"),
        "target_prediction_lockset_entry_count": dig(lockset, "entry_count"),
        "target_prediction_lockset_file": entry("reports/m10/TARGET_PREDICTION_LOCKSET.json"),
        "target_label_reveal_identity": dig(m10acc, "target_label_reveal_identity"),
        "target_label_reveal_file": entry("reports/m10/TARGET_LABEL_REVEAL.json"),
        "pre_reveal_audit_file": entry("reports/m10/PRE_REVEAL_AUDIT.json"),
    },
    "final_scientific_checkpoint": {
        "commit": git("rev-parse", "HEAD"),
        "tag": "m10-blind-evaluation-checkpoint",
        "tag_peeled": git("rev-parse", "m10-blind-evaluation-checkpoint^{}"),
        "tag_object": git("rev-parse", "m10-blind-evaluation-checkpoint"),
    },
    "version_b_test_baseline": {
        "command": dig(tests, "command"),
        "passed": dig(tests, "passed"),
        "failed": dig(tests, "failed"),
        "skipped": dig(tests, "skipped"),
        "errors": dig(tests, "errors"),
        "raw": dig(tests, "raw"),
        "file": entry("reports/m10/TEST_SUITE.json"),
    },
    "llm_gap_evidence": {
        "m7_generator_provider": dig(m7lock, "generator", "provider"),
        "m7_generator_model_id": dig(m7lock, "generator", "model_id"),
        "m7_generator_revision": dig(m7lock, "generator", "revision"),
        "m7_generator_external_llm_invoked": dig(m7lock, "generator", "external_llm_invoked"),
        "generator_json_file": entry("assets/recipe_banks/prism_recipe_bank_m7_v1/generator.json"),
        "generate_py_file": entry("src/prism_fas/recipes/generate.py"),
        "bank_py_file": entry("src/prism_fas/recipes/bank.py"),
        "bank_m7_yaml_file": entry("configs/recipes/bank_m7.yaml"),
        "random_operator_bank_py_file": entry("src/prism_fas/synthesis/random_operator_bank.py"),
    },
    "narrative": entry("reports/paper/PRISM_FAS_B_FULL_PROJECT_NARRATIVE.html"),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
               encoding="utf-8")
print("wrote", OUT)
print("head        :", snapshot["git"]["head"])
print("tag peeled  :", snapshot["git"]["tag_m10_blind_evaluation_checkpoint_peeled"])
print("clean tree  :", snapshot["git"]["status_clean"])
print("NOT FOUND   :", json.dumps(snapshot).count(NOT_FOUND))
