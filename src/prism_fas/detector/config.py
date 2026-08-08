"""Load the frozen M9 YAML configs into the typed dataclasses.

One loader, used identically by the local CPU smoke and by the Modal wrapper, so
the two can never drift into different hyper-parameters. Every identity the config
pins is re-checked against the real artifact before training starts.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import yaml
from .losses import DEFAULT_WEIGHTS
from .prism_detector import DetectorConfig
from .sampler import BatchContract
from .trainer import M9TrainingConfig, batch_contract_for
from .variant import ResolvedExperimentVariant

CONFIG_SCHEMA_VERSION = "m9-config-v1"


class ConfigError(ValueError):
    """An M9 config file does not satisfy the frozen contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise ConfigError(f"{Path(path).name} is not a mapping")
    return payload


def detector_config_from(payload: dict[str, Any],
                         variant: ResolvedExperimentVariant | None = None) -> DetectorConfig:
    """Build the detector config for one M10 variant.

    `variant` defaults to the frozen B08 reference, so every M9 call site keeps
    building exactly the M9 reference detector. `prototype_k` follows the variant
    when it declares a manifold, because the flag set is the authority on the
    science and the YAML carries the reference value.
    """
    resolved = (variant or ResolvedExperimentVariant.reference()).validate()
    model = payload["model"]
    backbones = payload["backbones"]
    manifold, prompt, fusion = model["manifold"], model["prompt"], model["fusion"]
    prototype_k = int(resolved.prototype_k) if resolved.has_manifold else int(manifold["prototype_k"])
    return DetectorConfig(
        variant=resolved,
        region_dim=int(model["region_embedding_dim"]),
        region_attention_heads=int(model["region_attention_heads"]),
        visibility_threshold=float(model["visibility_threshold"]),
        optional_highpass=bool(model["optional_highpass"]),
        local_model_name=str(backbones["local"]["timm_name"]),
        global_model_id=str(backbones["global"]["model_id"]),
        global_revision=str(backbones["global"]["revision"]),
        prototype_k=prototype_k,
        covariance_epsilon=float(manifold["covariance_epsilon"]),
        tau_prototype=float(manifold["tau_prototype"]),
        prototype_decay=float(manifold["update_decay"]),
        distance_scale_convention=str(manifold["distance_scale_convention"]),
        prompt_temperature=float(prompt["temperature"]),
        fusion_top_k=int(fusion["topk"]),
        global_head_dropout=float(model.get("global_head_dropout", 0.0)),
        region_order=tuple(str(name) for name in model["region_order"]))


def training_config_from(payload: dict[str, Any], *, run_id: str | None = None,
                         overrides: dict[str, Any] | None = None,
                         variant: ResolvedExperimentVariant | None = None) -> M9TrainingConfig:
    resolved = (variant or ResolvedExperimentVariant.reference()).validate()
    run, batch, optimizer = payload["run"], payload["batch"], payload["optimizer"]
    scheduler, stages, loss = payload["scheduler"], payload["stages"], payload["loss"]
    checkpoint, validation = payload["checkpoint"], payload["validation"]
    weights = {**DEFAULT_WEIGHTS, **{key: float(value) for key, value in loss["weights"].items()}}
    config = M9TrainingConfig(
        run_id=str(run_id or run["id"]), seed=int(run["seed"]),
        warmup_detector_epochs=int(stages["warmup_detector_epochs"]),
        manifold_warmup_epochs=int(stages["manifold_warmup_epochs"]),
        mixed_epochs=int(stages["mixed_epochs"]),
        steps_per_epoch=int(batch["steps_per_epoch"]),
        accumulation_steps=int(batch["accumulation_steps"]),
        backbone_lr=float(optimizer["backbone_lr"]), head_lr=float(optimizer["head_lr"]),
        weight_decay=float(optimizer["weight_decay"]), betas=tuple(float(v) for v in optimizer["betas"]),
        warmup_fraction=float(scheduler["warmup_fraction"]),
        min_lr_scale=float(scheduler["min_lr_scale"]),
        gradient_clip_norm=float(payload["gradient_clip_norm"]), amp=bool(payload["amp"]),
        ema_enabled=bool(payload["ema"]["enabled"]),
        checkpoint_every_steps=int(checkpoint["every_steps"]),
        validate_every_epochs=int(validation["every_epochs"]),
        validation_batch_size=int(validation["batch_size"]),
        selection_metric=str(checkpoint["selection_metric"]),
        tie_break_metric=str(checkpoint["tie_break_metric"]),
        calibration_metric=str(checkpoint["calibration_metric"]),
        loss_weights=weights, clean_cap=float(loss["clean_cap"]),
        mil_temperature=float(loss["mil_temperature"]),
        prompt_temperature=float(loss["prompt_temperature"]),
        prototype_seed=int(payload["prototypes"]["seed"]),
        prototype_batch_size=int(payload["prototypes"]["batch_size"]),
        variant=resolved)
    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
    return config


def batch_contract_from(payload: dict[str, Any],
                        variant: ResolvedExperimentVariant | None = None) -> BatchContract:
    """The G5 batch contract this variant declares.

    For the reference this is Table 36's 12/12/8. For a `synthetic: none` row it is
    the real-only composition: no fabricated synthetic slots, and no synthetic loss
    evaluating as a constant placeholder.
    """
    resolved = (variant or ResolvedExperimentVariant.reference()).validate()
    batch = payload["batch"]
    if not resolved.uses_synthetic or not resolved.domain_balance:
        from dataclasses import replace as _replace
        base = training_config_from(payload, variant=resolved)
        return batch_contract_for("G5", base)
    return BatchContract(real_live=int(batch["live"]), real_spoof=int(batch["real_spoof"]),
                         synthetic=int(batch["synthetic_spoof"]),
                         domain_balance=bool(batch["domain_balance"]),
                         require_both_routes=resolved.requires_both_routes,
                         routes=resolved.synthetic_routes,
                         accumulation_steps=int(batch["accumulation_steps"])).validate()


def verify_pinned_identities(model_payload: dict[str, Any], *, package_identity: str,
                             bank_identity: str, recipe_bank_identity: str,
                             siglip2_identity: str | None = None,
                             text_cache_identity: str | None = None) -> dict[str, Any]:
    """Fail before any work if a real artifact is not the pinned one.

    A config pin that is never checked is decoration; these are checked.
    """
    checks = {
        "source_package": (model_payload["source_package"]["expected_content_identity_sha256"], package_identity),
        "synthetic_bank": (model_payload["synthetic_bank"]["expected_content_identity_sha256"], bank_identity),
        "recipe_bank": (model_payload["recipe_bank"]["expected_identity_sha256"], recipe_bank_identity)}
    if siglip2_identity is not None:
        checks["siglip2"] = (model_payload["backbones"]["global"]["identity_sha256"], siglip2_identity)
    if text_cache_identity is not None:
        checks["recipe_text_cache"] = (model_payload["model"]["prompt"]["cache_identity_sha256"], text_cache_identity)
    mismatched = {name: {"config": pinned, "actual": actual}
                  for name, (pinned, actual) in checks.items() if pinned != actual}
    if mismatched: raise ConfigError(f"pinned identities do not match the real artifacts: {mismatched}")
    return {name: actual for name, (_, actual) in checks.items()}


def assert_no_target_paths(payload: dict[str, Any]) -> None:
    """No target path may appear anywhere in a training config."""
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    for needle in ("siw_mv2", "siw-mv2", "target_test/", "/target_test", "target_test\\"):
        if needle in text: raise ConfigError(f"a training config must not reference {needle!r}")


def assert_no_superseded_bank(payload: dict[str, Any]) -> None:
    """The v1 and v2 banks are superseded and must never be reachable."""
    text = json.dumps(payload, sort_keys=True, default=str)
    for needle in ("prism_synthetic_bank_m8_v1", "prism_synthetic_bank_m8_v2"):
        if needle in text: raise ConfigError(f"a training config must not reference {needle!r}")


def resolved_config(model_payload: dict[str, Any], training_payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": CONFIG_SCHEMA_VERSION, "model": model_payload, "training": training_payload}


def resolved_config_hash(model_payload: dict[str, Any], training_payload: dict[str, Any]) -> str:
    body = resolved_config(model_payload, training_payload)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def load_m9_configs(model_path: Path, training_path: Path,
                    variant: ResolvedExperimentVariant | None = None) -> dict[str, Any]:
    """Everything the trainer and the Modal wrapper need, from the two YAMLs.

    One loader for every M10 row: the YAMLs carry the shared, frozen
    hyper-parameters and `variant` carries the row's scientific switches. There is
    no second config file per baseline, because every baseline is a CONFIGURATION of
    the shared implementation.
    """
    resolved = (variant or ResolvedExperimentVariant.reference()).validate()
    model_payload = load_yaml(Path(model_path))
    training_payload = load_yaml(Path(training_path))
    assert_no_target_paths(training_payload)
    assert_no_target_paths(model_payload)
    assert_no_superseded_bank(training_payload)
    assert_no_superseded_bank(model_payload)
    return {"model_payload": model_payload, "training_payload": training_payload,
            "variant": resolved,
            "detector_config": detector_config_from(model_payload, resolved),
            "training_config": training_config_from(training_payload, variant=resolved),
            "batch_contract": batch_contract_from(training_payload, resolved),
            "resolved_config_hash": resolved_config_hash(model_payload, training_payload)}
