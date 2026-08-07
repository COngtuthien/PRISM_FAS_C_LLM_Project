"""M9 local CPU smoke on REAL data.

Exercises the whole declared path once, on a small real subset: the canonical
source loader, the frozen M8 v3 accepted bank, the 12/12/8 composition, region
priors and visibility, the cached SigLIP2 recipe text embeddings, PromptHead, the
global and regional detector, the Table 34 fusion, every declared loss, `q`
weighting, prototype initialization, forward, backward, an optimizer step,
checkpoint save and strict resume.

No random tensor stands in for a sample: every image is a real `source_train` crop
or a real accepted synthetic sample. `target_test` is never opened.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_fas.detector.config import load_m9_configs, verify_pinned_identities  # noqa: E402
from prism_fas.detector.dataset import batch_composition, domain_composition      # noqa: E402
from prism_fas.detector.trainer import M9Trainer, source_isolation_report         # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 local CPU smoke on real data")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--package-root", type=Path, default=root / "data/processed/prism_data_v1_m3b")
    parser.add_argument("--bank-root", type=Path,
                        default=root / "data/processed/prism_synthetic_bank_m8_v3_e84c78cd2a9b")
    parser.add_argument("--recipe-bank-root", type=Path,
                        default=root / "assets/recipe_banks/prism_recipe_bank_m7_v1")
    parser.add_argument("--weight-root", type=Path,
                        default=Path("D:/AI on IOT/Anti_spoofing/model_cache"))
    parser.add_argument("--run-root", type=Path, default=root / "runs/m9_local_cpu_smoke")
    parser.add_argument("--cache-root", type=Path, default=root / "data/processed/m9_cache")
    parser.add_argument("--report", type=Path, default=root / "reports/m9/local_cpu_smoke.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resume-steps", type=int, default=4)
    parser.add_argument("--prototype-batch-size", type=int, default=28)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.time()
    configs = load_m9_configs(root / "configs/models/m9_detector.yaml",
                              root / "configs/train/m9_reference.yaml")
    training = replace(configs["training_config"], run_id="m9_local_cpu_smoke",
                       steps_per_epoch=8, checkpoint_every_steps=0, amp=False,
                       prototype_batch_size=int(args.prototype_batch_size),
                       validation_limit=int(args.validation_limit))
    trainer = M9Trainer(config=training, detector_config=configs["detector_config"],
                        package_root=args.package_root, bank_root=args.bank_root,
                        recipe_bank_root=args.recipe_bank_root, run_root=args.run_root,
                        cache_root=args.cache_root, weight_root=args.weight_root,
                        loader_config_path=root / "configs/data/loader_m4.yaml",
                        device=args.device, validation_limit=int(args.validation_limit),
                        progress=lambda payload: print(f"  {payload}", flush=True))
    pins = verify_pinned_identities(
        configs["model_payload"], package_identity=trainer.dataset.package_identity,
        bank_identity=trainer.dataset.bank.identity, recipe_bank_identity=trainer.recipe_bank_identity,
        siglip2_identity=trainer.siglip.identity(), text_cache_identity=trainer.text_cache.identity)
    print(json.dumps({"pinned_identities_verified": pins}, indent=1), flush=True)

    # One mixed batch, inspected before any training: composition, visibility,
    # attack masks and q must all be exactly what the contract declares.
    plan = trainer.samplers["G5"].epoch_plans(0)[0]
    batch = trainer.dataset.batch_from_plan(plan)
    inspection = {
        "composition": batch_composition(batch), "domains": domain_composition(batch),
        "synthetic_routes": sorted({trainer.dataset.bank.rows[index]["route"] for index in plan.synthetic}),
        "region_priors_shape": list(batch.region_priors.shape),
        "visible_regions": int((batch.visibility >= trainer.detector_config.visibility_threshold).sum()),
        "attacked_regions": int(batch.attack_region_mask.sum()),
        "q_min": float(batch.quality_weight[batch.is_synthetic].min()),
        "q_max": float(batch.quality_weight[batch.is_synthetic].max()),
        "real_rows_carry_no_attack_mask": float(batch.attack_region_mask[~batch.is_synthetic].sum()) == 0.0,
        "every_synthetic_row_is_spoof": bool((batch.label[batch.is_synthetic] == 1).all())}
    print(json.dumps({"batch_inspection": inspection}, indent=1), flush=True)

    smoke = trainer.smoke(steps=int(args.steps), resume_steps=int(args.resume_steps))
    validation = trainer.validate()
    report = {
        "schema_version": "m9-local-cpu-smoke-v1", "device": args.device,
        "seconds": round(time.time() - started, 2),
        "pinned_identities": pins, "batch_inspection": inspection, "smoke": smoke,
        "source_dev_validation": validation,
        "dataset": trainer.dataset.summary(), "run_summary": trainer.run_summary(),
        "source_isolation": source_isolation_report(trainer, source_dev_opened=True),
        "passed": bool(smoke["all_losses_finite"] and not smoke["restarted_at_zero"]
                       and smoke["sampler_continued"]
                       and smoke["steps_after_resume"] > smoke["steps_before_checkpoint"]
                       and inspection["composition"] == {"real_live": 12, "real_spoof": 12,
                                                         "synthetic_spoof": 8}
                       and inspection["real_rows_carry_no_attack_mask"]
                       and all(math.isfinite(float(value)) for value in validation.values()))}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.report),
                      "seconds": report["seconds"]}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
