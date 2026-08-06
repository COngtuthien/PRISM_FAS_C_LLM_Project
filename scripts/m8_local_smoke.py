"""M8 local CPU smoke: real source_train pairs through the GPAT model and losses.

CPU fp32 only — full GPAT training runs on Modal L4, never here. No source_dev,
no target_test, no GPU lab and no SSH.
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import torch
from prism_fas.synthesis.dwt import DWT_CONVENTION, DWT_RECONSTRUCTION_TOLERANCE_FP32, reconstruction_error
from prism_fas.synthesis.gpat_contracts import LL_INVARIANT_TOLERANCE
from prism_fas.synthesis.gpat_losses import assert_invariants, compute_losses, loss_manifest
from prism_fas.synthesis.gpat_model import build_gpat_model
from prism_fas.synthesis.m8_pipeline import (SampleStore, SourceOnlyAudit, build_batch, config_hash,
                                             load_gpat_config, load_pairs, resolve_bank)
from prism_fas.synthesis.quality_models import QualityModelRegistry
from prism_fas.utils.core import atomic_json_write

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT / "data" / "processed" / "prism_data_v1_m3b")
    parser.add_argument("--bank", type=Path, default=ROOT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "synthesis" / "gpat_m8.yaml")
    parser.add_argument("--pairs", type=Path, default=ROOT / "reports" / "m8" / "pairs")
    parser.add_argument("--weight-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "m8" / "local_gpat_smoke.json")
    parser.add_argument("--pairs-count", type=int, default=2)
    args = parser.parse_args()

    weight_root = args.weight_root
    if weight_root is None:
        import yaml
        weight_root = Path(yaml.safe_load((ROOT / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))["model_cache"])

    config = load_gpat_config(args.config)
    torch.manual_seed(int(config["seed"]))
    audit = SourceOnlyAudit()
    store = SampleStore.open(args.package_root, audit)
    bank = resolve_bank(args.bank)
    pairs = load_pairs(args.pairs, "train")[: int(args.pairs_count)]
    registry = QualityModelRegistry.resolve(weight_root, roles=("identity",))
    identity = registry.adaface("cpu")

    fixture = torch.rand(2, 3, 224, 224)
    recon = reconstruction_error(fixture)
    model = build_gpat_model(config)
    batch = build_batch(store, pairs, bank, identity, device="cpu")
    output = model.forward_batch(batch)
    output.validate(batch.live_image)
    generated_embedding = identity(output.synthetic_image)
    result = compute_losses(output, batch, generated_embedding, config.get("loss"))
    assert_invariants(result, ll_tolerance=float(config["invariants"]["ll_max_abs_error"]))
    result.total.backward()
    gradients = {name: float(parameter.grad.abs().max().item())
                 for name, parameter in model.named_parameters() if parameter.grad is not None}
    identity_grads = [parameter.grad for parameter in identity.parameters() if parameter.grad is not None]

    payload = {
        "stage": "m8_local_gpat_smoke", "device": "cpu", "precision": "fp32", "modal_used": False,
        "gpu_used": False, "ssh_used": False, "torch": torch.__version__, "python": platform.python_version(),
        "package_root_name": Path(args.package_root).name, "bank_id": bank["bank_id"],
        "package_identity": json.loads((Path(args.package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"],
        "recipe_bank_identity": bank["lock"]["bank_content_identity_sha256"],
        "config_hash": config_hash(config),
        "dwt": {"convention": DWT_CONVENTION, "reconstruction_max_abs_error": recon,
                "tolerance": DWT_RECONSTRUCTION_TOLERANCE_FP32, "passed": recon <= DWT_RECONSTRUCTION_TOLERANCE_FP32},
        "model": {"architecture_hash": model.architecture_hash(), "parameter_count": model.parameter_count(),
                  "delta_ll_enabled": False, "max_high_frequency_delta": model.max_high_frequency_delta},
        "pairs": [{"pair_id": pair["pair_id"], "live_sample_id": pair["live_sample_id"],
                   "spoof_sample_id": pair["spoof_sample_id"], "recipe_id": pair["recipe_id"],
                   "domain_relation": pair["domain_relation"], "live_dataset": pair["live_dataset"],
                   "spoof_dataset": pair["spoof_dataset"]} for pair in pairs],
        "losses": result.detached(), "loss_manifest": loss_manifest(config.get("loss")),
        "invariants": {"ll_max_abs_error": result.metrics["ll_invariant_max_abs_error"],
                       "ll_tolerance": LL_INVARIANT_TOLERANCE,
                       "ll_passed": result.metrics["ll_invariant_max_abs_error"] <= LL_INVARIANT_TOLERANCE,
                       "outside_mask_max_abs_error": result.metrics["outside_mask_max_abs_error"],
                       "outside_mask_exactly_zero": result.metrics["outside_mask_max_abs_error"] == 0.0},
        "backward": {"finite": all(value == value and value != float("inf") for value in gradients.values()),
                     "parameters_with_gradient": len(gradients),
                     "max_abs_gradient": max(gradients.values()) if gradients else 0.0,
                     "identity_model_received_gradients": bool(identity_grads)},
        "quality_models": registry.manifest(),
        "source_isolation": audit.report()}
    payload["passed"] = bool(payload["dwt"]["passed"] and payload["invariants"]["ll_passed"]
                             and payload["invariants"]["outside_mask_exactly_zero"]
                             and payload["backward"]["finite"] and not identity_grads
                             and payload["source_isolation"]["source_dev_opened"] is False
                             and payload["source_isolation"]["target_test_opened"] is False)
    atomic_json_write(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("stage", "device", "passed", "losses", "invariants", "backward")},
                     indent=2, default=str))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
