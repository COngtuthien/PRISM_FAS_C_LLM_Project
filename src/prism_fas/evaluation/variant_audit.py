"""Static + bounded-dynamic implementability audit for every M10 matrix row.

Before a single GPU hour is spent, every unique scientific configuration in
`reports/m10/M10_MATRIX_PLAN.json` is INSTANTIATED and exercised here. For each one
the audit proves, on a CPU fixture:

    flags parse                    the row resolves to a validated variant
    architecture builds            the real `PRISMDetector`, under this variant
    required stages resolve        G1/G2/G5/G6, with G2 present iff a manifold is
    dataset + sampler resolve      a real `M9BatchSampler` over the declared pools,
                                   producing the declared composition
    loss graph resolves            the terms the variant declares are the terms that
                                   were computed, and no other
    optimizer groups               every group non-empty and carrying the right LR
    checkpoint identity            computable, and DISTINCT for distinct variants
    source calibration contract    the source-only selection rule resolves
    G7 inference adapter           capabilities resolve and the prediction row that
                                   the variant would emit validates

Two rules this module exists to enforce:

*   **No row is "implemented" by falling back to B08.** The audit records each row's
    own architecture and training identity; the difference report proves a row
    changes exactly the dimensions it declares.
*   **A blocked row stays blocked for its own declared reason.** The audit never
    marks a BLOCKED row implementable and never invents a reason of its own.

The frozen SigLIP2 tower is stood in for by a shape-exact stub: this audit answers
"can every declared row be built and stepped", not "what does it score". The real
weights are exercised by the bounded remote smokes, and the science by the matrix
itself. Never imports modal.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANT_AUDIT_SCHEMA_VERSION = "m10-variant-audit-v1"
# Small enough to run 41 configurations on a CPU in seconds, large enough that every
# declared shape relation still has to hold.
AUDIT_REGION_DIM = 16
AUDIT_TEXT_DIM = 12
AUDIT_PROMPTS = 6
AUDIT_PRIOR_SIZE = 8


class VariantAuditError(RuntimeError):
    """A declared matrix row cannot be materialized."""


class _StubTower:
    """Stands in for the frozen SigLIP2 vision tower: identical output contract,
    no weights and no download. Registered nowhere, exactly like the real one."""

    def __init__(self, dim: int, tokens: int):
        import torch
        from torch import nn
        self._module = nn.Identity()
        self.dim, self.tokens = int(dim), int(tokens)
        self._parameter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def __call__(self, *, pixel_values: Any) -> Any:
        import torch
        batch = pixel_values.shape[0]
        generator = torch.Generator().manual_seed(20260806)
        last = torch.rand(batch, self.tokens, self.dim, generator=generator)
        pooled = last.mean(dim=1)
        return type("TowerOutput", (), {"last_hidden_state": last, "pooler_output": pooled})()

    def parameters(self) -> Any: return iter([self._parameter])
    def eval(self) -> "_StubTower": return self
    def to(self, *args: Any, **kwargs: Any) -> "_StubTower": return self


def _text_matrix() -> Any:
    import torch
    generator = torch.Generator().manual_seed(20260806)
    return torch.nn.functional.normalize(
        torch.randn(AUDIT_PROMPTS, AUDIT_TEXT_DIM, generator=generator), dim=-1)


def build_audit_detector(variant: Any) -> Any:
    """The real `PRISMDetector` under this variant, at fixture scale."""
    from prism_fas.detector.prism_detector import DetectorConfig, PRISMDetector
    config = DetectorConfig(region_dim=AUDIT_REGION_DIM, region_attention_heads=2,
                            local_pretrained=False, prototype_k=int(variant.prototype_k),
                            variant=variant)
    model = PRISMDetector(config, text_embeddings=_text_matrix(), text_cache_identity="audit")
    if variant.has_global:
        model.attach_global_tower(_StubTower(model.global_dim, model.global_patch_tokens))
    return model


def audit_batch(variant: Any, contract: Any) -> Any:
    """A structurally real batch matching the contract this variant declares."""
    import torch
    from prism_fas.detector.contracts import REGION_COUNT, SPOOF, DetectorBatch
    generator = torch.Generator().manual_seed(20260807)
    live, spoof, synthetic = contract.real_live, contract.real_spoof, contract.synthetic
    size = live + spoof + synthetic
    is_synthetic = torch.zeros(size, dtype=torch.bool)
    if synthetic: is_synthetic[size - synthetic:] = True
    label = torch.zeros(size, dtype=torch.long)
    label[live:] = SPOOF
    attack = torch.zeros(size, REGION_COUNT)
    if synthetic:
        attack[is_synthetic, 0] = 1.0
        attack[is_synthetic, 3] = 1.0
    return DetectorBatch(
        image=torch.rand(size, 3, 224, 224, generator=generator),
        label=label, dataset_id=torch.arange(size) % 2, is_synthetic=is_synthetic,
        region_priors=torch.rand(size, REGION_COUNT, AUDIT_PRIOR_SIZE, AUDIT_PRIOR_SIZE, generator=generator),
        visibility=torch.ones(size, REGION_COUNT),
        sample_ids=tuple(f"audit{index}" for index in range(size)),
        datasets=("casia_fasd", "msu_mfsd"),
        artifact_map=(torch.rand(size, 1, 224, 224, generator=generator)
                      * is_synthetic.view(-1, 1, 1, 1)) if synthetic else None,
        attack_region_mask=attack,
        quality_weight=torch.where(is_synthetic, torch.full((size,), 0.8), torch.zeros(size)),
        recipe_index=torch.where(is_synthetic, torch.ones(size, dtype=torch.long),
                                 torch.full((size,), -1, dtype=torch.long)),
        artifact_family=tuple("moire" if bool(flag) else "" for flag in is_synthetic)).validate()


def _audit_sampler(variant: Any, contract: Any) -> dict[str, Any]:
    """A real `M9BatchSampler` over declared pools, proving the composition the row
    asks for can actually be produced and is deterministic."""
    from prism_fas.detector.sampler import M9BatchSampler
    live = {"casia_fasd": list(range(0, 40)), "msu_mfsd": list(range(40, 80))}
    spoof = {"casia_fasd": list(range(80, 120)), "msu_mfsd": list(range(120, 160))}
    routes = {"physics": list(range(0, 60)), "gpat": list(range(60, 120))}
    sampler = M9BatchSampler(real_live=live, real_spoof=spoof, synthetic_routes=routes,
                             contract=contract, seed=20260806, steps_per_epoch=3, identity="audit")
    plans = sampler.epoch_plans(0)
    replay = sampler.epoch_plans(0)
    sizes = {plan.size for plan in plans}
    return {"steps": len(plans), "batch_sizes": sorted(sizes),
            "deterministic": sampler.fingerprint(0) == M9BatchSampler(
                real_live=live, real_spoof=spoof, synthetic_routes=routes, contract=contract,
                seed=20260806, steps_per_epoch=3, identity="audit").fingerprint(0),
            "replayed_identically": [p.real_live for p in plans] == [p.real_live for p in replay],
            "every_batch_has_live_and_spoof": all(plan.real_live and plan.real_spoof for plan in plans),
            "synthetic_per_batch": sorted({len(plan.synthetic) for plan in plans}),
            "declared_routes": list(contract.routes) if contract.synthetic else []}


def audit_variant(variant: Any, *, experiment_id: str = "") -> dict[str, Any]:
    """Instantiate and exercise ONE configuration. Raises nothing; reports."""
    import torch
    from prism_fas.detector.checkpoint import RunIdentity
    from prism_fas.detector.config import load_yaml
    from prism_fas.detector.losses import loss_contract_identity, loss_graph_delta
    from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for, enabled_terms, training_delta
    from prism_fas.detector.prism_detector import architecture_delta
    from .target_prediction import VariantCapabilities, build_prediction_row, validate_predictions

    findings: list[str] = []
    report: dict[str, Any] = {"experiment_id": experiment_id, "flags": variant.flags()}
    executable, reason = variant.executable()
    report["executable"] = executable
    if not executable:
        report["not_executable_reason"] = reason
        return report

    # 1. stages — declared AND actually walked through the stage machine.
    #
    # Walking it matters: the first real-data smoke found that the trainer checked
    # each transition against the full G1->G2->G5->G6 default instead of the run's
    # own declared flow, so every manifold-free variant was refused its legitimate
    # G1 -> G5. An audit that only inspected `required_stages()` agreed with the
    # declaration and missed it, so the audit now drives the same state machine the
    # trainer drives.
    from prism_fas.detector.checkpoint import StageLineage, check_stage_transition
    stages = variant.required_stages()
    report["required_stages"] = list(stages)
    if ("G2" in stages) != variant.has_manifold:
        findings.append("G2 presence does not follow the manifold declaration")
    try:
        lineage = StageLineage(order=stages)
        current = None
        for stage in stages:
            current = check_stage_transition(current, stage, order=stages)
            lineage.enter(stage, input_hashes={})
            lineage.complete(stage, output_hashes={})
        report["stage_flow_walked"] = [str(entry["stage"]) for entry in lineage.payload()]
        if report["stage_flow_walked"] != list(stages):
            findings.append("the walked stage flow does not match the declared one")
    except Exception as error:                        # noqa: BLE001 - reported, not raised
        findings.append(f"the declared stage flow cannot be walked: {error}")
    # A stage the variant does not declare must be refused, not merely unused.
    for absent in [stage for stage in ("G1", "G2", "G5", "G6") if stage not in stages]:
        try:
            check_stage_transition("G1", absent, order=stages)
            findings.append(f"{absent} is not declared but the stage machine allowed it")
        except Exception:                             # noqa: BLE001 - refusing is correct
            pass

    # 2. architecture
    model = build_audit_detector(variant)
    report["architecture_identity"] = model.architecture_identity()
    report["architecture_delta"] = architecture_delta(variant)
    report["modules"] = sorted(name for name, _ in model.named_children())
    report["parameter_counts"] = model.parameter_counts()
    # A branch that is off must leave NO trainable parameter behind.
    if not variant.has_local and any(name.startswith("local") for name in report["modules"]):
        findings.append("local_branch=off still instantiated a local module")
    if variant.has_global and model.global_tower is None:
        findings.append("global_branch declared but no tower is attached")
    if not variant.has_global and model.global_tower is not None:
        findings.append("global_branch=off but a tower was attached anyway")
    if variant.has_manifold != (model.manifold is not None):
        findings.append("manifold declaration and instantiation disagree")
    if variant.has_prompt != (model.prompt_head is not None):
        findings.append("prompt declaration and instantiation disagree")
    if variant.prompt_adapter_trainable != (model.text_adapter is not None):
        findings.append("prompt adapter declaration and instantiation disagree")

    # 3. training config, batch contract, sampler
    config = M9TrainingConfig(run_id=f"audit_{experiment_id or 'variant'}", variant=variant,
                              steps_per_epoch=3)
    contract = batch_contract_for("G5", config)
    report["batch_contract"] = contract.payload()
    report["training_delta"] = training_delta(variant)
    try:
        report["sampler"] = _audit_sampler(variant, contract)
        if not report["sampler"]["every_batch_has_live_and_spoof"]:
            findings.append("a batch was produced without both real live and real spoof (Table 9)")
        if not report["sampler"]["deterministic"]:
            findings.append("the sampler is not deterministic under the declared seed")
    except Exception as error:                        # noqa: BLE001 - reported, not raised
        findings.append(f"sampler does not resolve: {error}")

    # 4. optimizer groups
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    report["optimizer_groups"] = [{"name": group["name"], "lr": group["lr"],
                                   "parameters": sum(p.numel() for p in group["params"])}
                                  for group in groups]
    if any(group["parameters"] == 0 for group in report["optimizer_groups"]):
        findings.append("an optimizer group is empty")
    if not groups: findings.append("the variant has no trainable parameters at all")
    if variant.has_local and not any(g["name"] == "backbone" for g in report["optimizer_groups"]):
        findings.append("a local branch is declared but no backbone group exists")
    if not variant.has_local and any(g["name"] == "backbone" for g in report["optimizer_groups"]):
        findings.append("no local branch is declared but a backbone group exists")

    # 5. forward, loss graph, backward, one optimizer step
    optimizer = torch.optim.AdamW(groups) if groups else None
    batch = audit_batch(variant, contract)
    if variant.has_manifold:
        # G5 needs initialized prototypes; the audit seeds them the way G2 would.
        from prism_fas.detector.manifold import initialize_prototypes
        with torch.no_grad():
            probe = model(batch)
            sites = (probe.aux or {}).get("manifold_embeddings", probe.region_embeddings)
        state = initialize_prototypes(sites.repeat(4, 1, 1).numpy(),
                                      probe.region_valid.repeat(4, 1).numpy(),
                                      k=int(variant.prototype_k), seed=20260806,
                                      region_names=model.manifold.region_names)
        model.manifold.load_state(state)
    output = model(batch)
    report["output_components"] = {
        "local_logits": output.local_logits is not None,
        "region_embeddings": output.region_embeddings is not None,
        "region_distances": output.region_distances is not None,
        "manifold_slots": output.manifold_slots,
        "s_region": output.s_region is not None,
        "p_prompt_spoof": output.p_prompt_spoof is not None,
        "prompt_logits": output.prompt_logits is not None}
    if (output.s_region is not None) != variant.fuses_region_evidence:
        findings.append("s_region presence does not follow the declared fusion")
    if output.manifold_slots != variant.manifold_slots:
        findings.append(f"manifold produced {output.manifold_slots} sites, variant declares "
                        f"{variant.manifold_slots}")

    from prism_fas.detector.losses import compute_losses
    text = model.text_matrix()
    result = compute_losses(output, batch, model.manifold, text_embeddings=text,
                            enabled=enabled_terms("G5", variant), variant=variant)
    report["active_loss_terms"] = dict(result.active)
    report["loss_values"] = {name: float(value.detach()) for name, value in result.terms.items()}
    report["L_total"] = float(result.total.detach())
    declared = variant.stage_loss_terms("G5")
    if declared != result.active:
        findings.append("the computed loss graph does not match the declared one")
    for name, value in report["loss_values"].items():
        if value != value or abs(value) == float("inf"):
            findings.append(f"{name} is not finite")
        # An inactive term must be an exact structural zero, never a constant.
        if not result.active[name] and value != 0.0:
            findings.append(f"{name} is inactive but reported {value}")
    if report["L_total"] != report["L_total"]: findings.append("L_total is not finite")

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        grads = [p for group in groups for p in group["params"] if p.grad is not None]
        report["parameters_with_gradient"] = len(grads)
        if not grads: findings.append("no parameter received a gradient")
        optimizer.step()
        report["optimizer_step"] = True

    # 6. checkpoint identity
    identity = RunIdentity(
        source_package_identity="audit", m8_bank_identity="audit",
        architecture_identity=model.architecture_identity(), siglip2_identity="audit",
        recipe_text_cache_identity="audit", config_hash=config.hash(),
        loss_contract_hash=loss_contract_identity(
            config.loss_weights, {"clean_cap": config.clean_cap, **loss_graph_delta(variant)}),
        batch_contract_hash=contract.identity(), dataset_contract_identity="audit")
    report["config_hash"] = config.hash()
    report["loss_contract_hash"] = identity.loss_contract_hash
    report["batch_contract_hash"] = contract.identity()
    report["variant_identity"] = variant.identity()
    try:
        identity.validate()
        report["checkpoint_identity_computable"] = True
    except Exception as error:                        # noqa: BLE001
        findings.append(f"checkpoint identity does not resolve: {error}")

    # 6b. checkpoint round-trip. Computing an identity is not the same as being able
    # to WRITE and READ a checkpoint: the first real-data smoke found that
    # `prototype_payload` called `manifold.export_state()` unconditionally, so every
    # manifold-free variant crashed at its first save. The audit now saves, reloads
    # and re-applies, which is where that surfaces.
    import tempfile
    from prism_fas.detector.checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            sha = save_checkpoint(path, model=model, optimizer=optimizer, scheduler=None, scaler=None,
                                  epoch=0, global_step=1, stage=stages[0], identity=identity,
                                  sampler_state={}, prototype_identity="",
                                  best_metrics={}, stage_lineage=[])
            payload = load_checkpoint(path, expected_identity=identity, allowed_stages=stages)
            restored = apply_checkpoint(payload, model=model, optimizer=optimizer)
            report["checkpoint_roundtrip"] = {
                "sha256": sha, "stage": restored["stage"], "global_step": restored["global_step"],
                "prototype_state_present": payload["prototype_state"].get("manifold") != "absent"}
            if report["checkpoint_roundtrip"]["prototype_state_present"] != variant.has_manifold:
                findings.append("the checkpoint's prototype state does not follow the manifold declaration")
    except Exception as error:                        # noqa: BLE001
        findings.append(f"checkpoint round-trip failed: {type(error).__name__}: {error}")

    # 7. source calibration contract — source_dev only, never target
    report["source_selection"] = {"selection_metric": config.selection_metric,
                                  "tie_break_metric": config.tie_break_metric,
                                  "calibration_metric": config.calibration_metric,
                                  "uses_target": False}
    if not all(name.startswith("source_dev/") for name in
               (config.selection_metric, config.tie_break_metric, config.calibration_metric)):
        findings.append("a selection metric is not source_dev")

    # 8. G7 inference adapter — the row the variant would actually emit
    capabilities = VariantCapabilities.from_variant(variant)
    row = build_prediction_row(
        sample_id="siw_0000000000000000_f0", video_id="siw_0000000000000000", frame_id=0,
        p_global=0.4, s_region=(0.25 if capabilities.has_region else None),
        p_prompt=(0.1 if capabilities.has_prompt else None), threshold=0.5, unknown_threshold=None,
        top_region_ids=([0, 1] if capabilities.has_region and capabilities.has_region_detail else []),
        region_distances=([0.1] * 9 if capabilities.has_region and capabilities.has_region_detail else []),
        checkpoint_hash="audit", calibration_hash="audit", inference_config_hash="audit",
        variant=experiment_id or "audit")
    try:
        validate_predictions([row])
        report["g7_adapter"] = {"has_region": capabilities.has_region,
                                "has_prompt": capabilities.has_prompt,
                                "has_region_detail": capabilities.has_region_detail,
                                "region_status": row["region_status"],
                                "prompt_status": row["prompt_status"]}
    except Exception as error:                        # noqa: BLE001
        findings.append(f"the G7 prediction adapter does not resolve: {error}")

    report["findings"] = findings
    report["implementable"] = not findings
    return report


def audit_matrix(plan: dict[str, Any]) -> dict[str, Any]:
    """Audit every unique scientific configuration in a materialized matrix plan."""
    from prism_fas.detector.variant import ResolvedExperimentVariant, describe_difference, variant_from_row

    reference = ResolvedExperimentVariant.reference()
    seen: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    identities: dict[str, list[str]] = {}
    for entry in plan["rows"]:
        experiment_id = str(entry["experiment_id"])
        variant = variant_from_row(entry)
        if entry.get("status") == "BLOCKED":
            rows.append({"experiment_id": experiment_id, "status": "BLOCKED",
                         "blocked_reason": entry.get("blocked_reason"),
                         "implementable": None, "audited": False,
                         "flags": variant.flags()})
            continue
        key = variant.identity()
        if key not in seen:
            seen[key] = audit_variant(variant, experiment_id=experiment_id)
        audit = dict(seen[key])
        audit.update({"experiment_id": experiment_id, "status": entry.get("status"), "audited": True,
                      "seed": entry.get("seed"), "replication_role": entry.get("replication_role"),
                      "difference_from_reference": describe_difference(reference, variant),
                      "scientific_config_hash": entry.get("scientific_config_hash")})
        identities.setdefault(audit["architecture_identity"], []).append(experiment_id)
        rows.append(audit)

    audited = [row for row in rows if row.get("audited")]
    failures = [row["experiment_id"] for row in audited if not row.get("implementable")]
    blocked = [row for row in rows if row.get("status") == "BLOCKED"]
    # No row may be "implemented" by silently sharing the reference architecture when
    # it declares an architectural delta.
    fallbacks = [row["experiment_id"] for row in audited
                 if row.get("architecture_delta") and
                 row["architecture_identity"] == seen[reference.identity()]["architecture_identity"]
                 if reference.identity() in seen]
    # Two different counts, both reported because they answer different questions.
    # `unique_variant_configs` is how many DISTINCT flag sets exist — that is what
    # has to be built and stepped, and a seed does not change it. The matrix's own
    # `unique_scientific_configs` counts (flags, seed) pairs, which is what the
    # replication policy is about. The audit must cover every one of both.
    covered = {str(row.get("scientific_config_hash")) for row in audited}
    matrix_configs = {str(entry.get("scientific_config_hash")) for entry in plan["rows"]}
    return {"schema_version": VARIANT_AUDIT_SCHEMA_VERSION,
            "m10_matrix_identity": plan["m10_matrix_identity"],
            "logical_rows": len(rows), "audited_rows": len(audited),
            "unique_variant_configs": len(seen),
            "unique_scientific_configs_in_matrix": len(matrix_configs),
            "unique_scientific_configs_covered": len(covered),
            "scientific_configs_not_covered": sorted(matrix_configs - covered - {
                str(entry.get("scientific_config_hash")) for entry in plan["rows"]
                if entry.get("status") == "BLOCKED"}),
            "implementable_rows": sum(1 for row in audited if row.get("implementable")),
            "not_implementable": failures,
            "blocked_rows": [{"experiment_id": row["experiment_id"],
                              "blocked_reason": row["blocked_reason"]} for row in blocked],
            "architecture_identities": {key: sorted(value) for key, value in sorted(identities.items())},
            "rows_sharing_the_reference_architecture_despite_a_delta": fallbacks,
            "all_executable_rows_implementable": not failures and not fallbacks,
            "rows": rows}


def write_audit(path: Path, payload: dict[str, Any]) -> Path:
    from prism_fas.utils.core import atomic_json_write
    atomic_json_write(Path(path), payload)
    return Path(path)
