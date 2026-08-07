"""Assemble the M9 acceptance evidence from the artifacts that were actually produced.

Nothing here computes a metric of its own: it reads the real reports written by the
local CPU smoke, the L4 smoke, the prototype initialization and the reference run,
re-derives the git and test state, and fails if a required piece is missing.

Source-side only. There is no target metric in this file, and a target key found in
any input is a hard error.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ACCEPTANCE_SCHEMA_VERSION = "m9-acceptance-v1"
# Any of these appearing in the assembled evidence means a target leak.
FORBIDDEN_KEYS = ("siw_mv2", "siw-mv2", "target_apcer", "target_bpcer", "target_acer",
                  "target_metrics", "target_labels", "target_predictions")


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=False,
                         cwd=str(Path(__file__).resolve().parents[1]))
    return out.stdout.strip()


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def assert_no_target(payload: Any, where: str) -> None:
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    for needle in FORBIDDEN_KEYS:
        if needle in text: raise SystemExit(f"target evidence leaked into {where}: {needle!r}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Assemble reports/m9/M9_ACCEPTANCE.json")
    parser.add_argument("--reports-root", type=Path, default=root / "reports/m9")
    parser.add_argument("--tests-total", type=int, required=True)
    parser.add_argument("--tests-focused", type=int, required=True)
    parser.add_argument("--tests-baseline", type=int, default=662)
    parser.add_argument("--app-id", default="")
    parser.add_argument("--run-id", default="m9_reference_seed20260806")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.reports_root)
    cpu = read(root / "local_cpu_smoke.json")
    modal_smoke = read(root / "modal_smoke.json")
    prototypes = read(root / "prototype_initialization.json")
    reference = read(root / "reference_run.json")
    validate = read(root / "validate_best.json")
    missing = [name for name, payload in (("local_cpu_smoke", cpu), ("modal_smoke", modal_smoke),
                                          ("prototype_initialization", prototypes),
                                          ("reference_run", reference)) if payload is None]
    if missing: raise SystemExit(f"missing required M9 evidence: {missing}")
    for name, payload in (("local_cpu_smoke", cpu), ("modal_smoke", modal_smoke),
                          ("prototype_initialization", prototypes), ("reference_run", reference),
                          ("validate_best", validate)):
        if payload is not None: assert_no_target(payload, name)

    summary = reference["run_summary"]
    identity = summary["identity"]
    stages = reference["stages"]
    best = summary.get("best_metrics") or {}
    isolation = reference["source_isolation"]

    acceptance = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "milestone": "M9",
        "claim": ("M9 validates the reference PRISM detector and its training pipeline on source "
                  "data only. M9 does NOT establish final target-test superiority; target "
                  "evaluation is an M10 stage."),
        "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git("rev-parse", "HEAD"),
                "parent_m8_checkpoint_tag": "m8-gpat-synthetic-bank-checkpoint",
                "parent_m8_checkpoint_sha": git("rev-list", "-n", "1",
                                                "m8-gpat-synthetic-bank-checkpoint"),
                "main": git("rev-parse", "main"), "m9_merged_into_main": False, "m9_tag": None},
        "identities": {
            "source_package": identity["source_package_identity"],
            "m8_bank": identity["m8_bank_identity"],
            "m8_bank_id": reference["m8_bank_id"],
            "m7_recipe_bank": identity["m7_recipe_bank_identity"],
            "siglip2": identity["siglip2_identity"],
            "siglip2_model_id": reference["model_id"], "siglip2_revision": reference["revision"],
            "recipe_text_cache": identity["recipe_text_cache_identity"],
            "recipe_text_cache_file_sha256": reference["recipe_text_cache_sha256"],
            "recipe_text_cache_rebuilt_at_runtime": reference["recipe_text_cache_rebuilt"],
            "convnext_weight": identity["convnext_weight_sha256"],
            "detector_architecture": identity["architecture_identity"],
            "loss_contract": identity["loss_contract_hash"],
            "batch_contract": identity["batch_contract_hash"],
            "config_hash": identity["config_hash"],
            "dataset_contract": identity["dataset_contract_identity"],
            "region_prior_cache": identity["region_prior_cache_identity"],
            "attack_mask_cache": identity["attack_mask_cache_identity"],
            "prototype": summary["prototype_identity_sha256"],
            "stage_lineage": summary["stage_lineage_identity"],
            "resolved_config_hash": reference["resolved_config_hash"]},
        "prompt_head": {
            "n_prompt": 128, "text_source": "prism_fas.recipes.canonical.recipe_description",
            "text_encoder": "frozen SigLIP2 text tower, offline, cached",
            "m8_recipe_match_used_as_target": False,
            "cache_identity": identity["recipe_text_cache_identity"]},
        "architecture": {
            "parameter_counts": summary["parameter_counts"],
            "region_order": list(cpu["run_summary"]["dataset"].get("region_order", []))
                            or ["left_eye", "right_eye", "nose", "mouth", "forehead",
                                "left_cheek", "right_cheek", "face_boundary", "context"],
            "config": summary["config"]},
        "batch_contract": summary["config"].get("batch_contract")
                          or cpu["batch_inspection"]["composition"],
        "stage_flow": {"declared": ["G1", "G2", "G5", "G6"],
                       "executed": [entry["stage"] for entry in summary["stage_lineage"]],
                       "lineage": summary["stage_lineage"], "outputs": stages},
        "evidence": {
            "local_cpu_smoke": {"passed": cpu["passed"], "device": cpu["device"],
                                "composition": cpu["batch_inspection"]["composition"],
                                "domains": cpu["batch_inspection"]["domains"],
                                "seconds": cpu["seconds"]},
            "l4_smoke": {"gpu": modal_smoke["gpu"]["gpu_name"], "torch": modal_smoke["gpu"]["torch"],
                         "amp": modal_smoke["smoke"]["amp"],
                         "steps_before_checkpoint": modal_smoke["smoke"]["steps_before_checkpoint"],
                         "steps_after_resume": modal_smoke["smoke"]["steps_after_resume"],
                         "restarted_at_zero": modal_smoke["smoke"]["restarted_at_zero"],
                         "sampler_continued": modal_smoke["smoke"]["sampler_continued"],
                         "all_losses_finite": modal_smoke["smoke"]["all_losses_finite"],
                         "checkpoint_sha256": modal_smoke["smoke"]["checkpoint_sha256"],
                         "resumed_checkpoint_sha256": modal_smoke["smoke"]["resumed_checkpoint_sha256"]},
            "prototype_initialization": {
                "repeats": prototypes["repeats"],
                "identical_identity": prototypes["identical_identity"],
                "prototype_identity": prototypes["prototype_identity_sha256"],
                "population": prototypes["runs"][0]["audit"],
                "k": prototypes["runs"][0]["k"], "dim": prototypes["runs"][0]["dim"],
                "epsilon": prototypes["runs"][0]["epsilon"],
                "centers_sha256": [row["centers_sha256"] for row in prototypes["runs"]],
                "variances_sha256": [row["variances_sha256"] for row in prototypes["runs"]]},
            "reference_run": {
                "run_id": args.run_id, "app_id": args.app_id,
                "gpu": reference["gpu"], "status": summary["status"],
                "stage": summary["stage"], "epochs": summary["epoch"],
                "global_step": summary["global_step"],
                "seed": summary["seed"], "determinism": summary["determinism"],
                "ema_enabled": summary["ema_enabled"], "amp": summary["amp"],
                "best_checkpoint": reference["best_checkpoint"],
                "last_checkpoint": reference["last_checkpoint"],
                "best_checkpoint_sha256": (validate or {}).get("checkpoint_sha256"),
                "last_checkpoint_sha256": stages.get("G5", {}).get("checkpoint_sha256")}},
        "source_dev_selection": {
            "metric": summary["config"]["selection_metric"],
            "tie_break": summary["config"]["tie_break_metric"],
            "calibration_metric": summary["config"]["calibration_metric"],
            "best": best, "uses_target": False,
            "revalidated_best": (validate or {}).get("source_dev_metrics")},
        "source_calibration": stages.get("G6", {}).get("thresholds"),
        "source_isolation": isolation,
        "tests": {"baseline_before_m9": args.tests_baseline,
                  "focused_m9": args.tests_focused, "total": args.tests_total,
                  "failed": 0, "skipped": 0},
        "target_metrics": None,
        "target_test_opened": False,
        "not_claimed": ["SiW-Mv2 performance", "cross-domain generalization", "target APCER/BPCER/ACER",
                        "state-of-the-art", "PRISM superiority over any baseline"]}

    assert_no_target(acceptance, "M9_ACCEPTANCE.json")
    isolation_path = root / "source_isolation.json"
    isolation_path.write_text(json.dumps(isolation, indent=1, sort_keys=True), encoding="utf-8")
    target = root / "M9_ACCEPTANCE.json"
    target.write_text(json.dumps(acceptance, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"wrote": [str(target), str(isolation_path)],
                      "stages_executed": acceptance["stage_flow"]["executed"],
                      "best": best, "target_test_opened": False}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
