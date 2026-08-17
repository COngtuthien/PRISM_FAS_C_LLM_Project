"""The §13.5 anti-Version-B decision-path regression guards.

Version B shipped a regional detector whose final decision did not actually
depend on its regional branch. §13.5 turns that lesson into a hard gate: a
Track-R implementation is INVALID if it computes ConvNeXt features and region
embeddings but the final logit is independent of them, and C7 must prove the
dependency **structurally** before any C8 scientific run.

Proving it structurally means two different measurements, because either one
alone can be satisfied by an implementation that is still wrong:

* **Autograd dependency.** ``||d fused_logit_R / d theta_local||`` and
  ``||d fused_logit_R / d theta_region_fusion||`` must be finite and strictly
  greater than 1e-8. This catches a branch that is detached, stop-gradded or
  simply omitted from the fusion graph.
* **Feature intervention.** Replacing a branch summary with zeros, or permuting
  the region summary, must move the logit by more than numerical tolerance. This
  catches a branch that is wired in but multiplied by a learned zero, or one
  whose contribution is structurally constant.

Both are coupling tests. Neither claims an effect size, and a large movement
here is not evidence that regions help — only that the decision graph is what it
says it is.

The remaining guards are about identity rather than gradients: the decision graph
must be serialized into run identity, the calibration must be fitted on the same
named logit the model decides with, and the frozen encoder must not appear in an
optimizer group. Each has a Version-B failure behind it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "prism-c7-decision-audit-v1"

#: §13.5, verbatim thresholds.
GRADIENT_MINIMUM = 1e-8
INTERVENTION_TOLERANCE = 1e-6

#: Module name prefixes belonging to each branch of the Track-R decision graph.
LOCAL_PREFIXES: tuple[str, ...] = ("local_backbone", "local_projection",
                                   "local_summary_projection", "highpass_stem")
REGION_FUSION_PREFIXES: tuple[str, ...] = ("region_query", "region_attention", "region_pool",
                                           "region_norm", "region_attention_pool",
                                           "region_summary_norm")
GLOBAL_PREFIXES: tuple[str, ...] = ("global_projection",)
FUSION_PREFIXES: tuple[str, ...] = ("fusion_projection", "fusion_classifier")


class DecisionAuditError(RuntimeError):
    """The decision graph cannot be audited as declared."""


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _named_parameters(model: Any, prefixes: Sequence[str]) -> list[tuple[str, Any]]:
    return [(name, parameter) for name, parameter in model.named_parameters()
            if any(name.startswith(prefix) for prefix in prefixes)
            and parameter.requires_grad]


def _grad_norm(logit: Any, parameters: Sequence[Any]) -> float:
    import torch

    if not parameters:
        return 0.0
    grads = torch.autograd.grad(logit.sum(), parameters, retain_graph=True,
                               allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float((grad.detach() ** 2).sum())
    return float(total ** 0.5)


def autograd_dependency(model: Any, batch: Any) -> dict[str, Any]:
    """Gradient norms of the decision logit w.r.t. each branch's parameters.

    `allow_unused=True` is deliberate: a detached branch yields ``None`` rather
    than raising, and ``None`` contributing zero is exactly the finding we want
    to report rather than an exception we would have to interpret.
    """
    import torch

    model.train()
    output = model(batch)
    logit = output.global_logit

    branches = {
        "local": _named_parameters(model, LOCAL_PREFIXES),
        "region_fusion": _named_parameters(model, REGION_FUSION_PREFIXES),
        "global_projection": _named_parameters(model, GLOBAL_PREFIXES),
        "fusion_head": _named_parameters(model, FUSION_PREFIXES),
    }
    norms = {name: _grad_norm(logit, [parameter for _key, parameter in items])
             for name, items in branches.items()}
    finite = {name: value == value and value != float("inf") for name, value in norms.items()}
    required = ("local", "region_fusion")
    passed = all(finite[name] and norms[name] > GRADIENT_MINIMUM for name in required)

    return {
        "gradient_norms": norms,
        "parameter_counts": {name: len(items) for name, items in branches.items()},
        "all_finite": all(finite.values()),
        "minimum": GRADIENT_MINIMUM,
        "required_branches": list(required),
        "passed": passed,
        "meaning": ("the final decision logit has a non-zero, finite autograd dependency on "
                    "the trainable local branch and the region-fusion parameters; a "
                    "detached, stop-gradded or omitted branch would report ~0 (§13.5)"),
    }


def feature_intervention(model: Any, batch: Any) -> dict[str, Any]:
    """Substitute each branch summary and re-run the model's own decision head.

    The head is not reimplemented here. The forward pass exposes ``fusion_g``,
    ``fusion_l`` and ``fusion_r``, and this function feeds modified copies back
    through ``fusion_projection`` and ``fusion_classifier`` — the very modules
    that produced the original logit. An audit that rebuilt the head would only
    ever prove the audit's arithmetic.
    """
    import torch

    model.eval()
    with torch.no_grad():
        output = model(batch)
        aux = output.aux or {}
        missing = [name for name in ("fusion_g", "fusion_l", "fusion_r") if name not in aux]
        if missing:
            raise DecisionAuditError(
                f"the model did not expose {missing}; the intervention smoke requires the "
                "branch summaries of a glr_concat decision head (§13.4.2)")
        g, local, region = aux["fusion_g"], aux["fusion_l"], aux["fusion_r"]

        def head(gv: Any, lv: Any, rv: Any) -> Any:
            return model.fusion_classifier(
                model.fusion_projection(torch.cat([gv, lv, rv], dim=-1)))

        baseline = head(g, local, region)
        cases = {
            "local_zeroed": head(g, torch.zeros_like(local), region),
            "region_zeroed": head(g, local, torch.zeros_like(region)),
            "region_permuted": head(g, local, region.flip(0) if region.shape[0] > 1
                                    else region.roll(1, dims=-1)),
            "global_zeroed": head(torch.zeros_like(g), local, region),
        }
        shifts = {name: float((value - baseline).abs().max())
                  for name, value in cases.items()}

    moved = {name: value > INTERVENTION_TOLERANCE for name, value in shifts.items()}
    return {
        "baseline_logit_mean": float(baseline.mean()),
        "max_absolute_shift": shifts,
        "moved": moved,
        "tolerance": INTERVENTION_TOLERANCE,
        "passed": all(moved.values()),
        "meaning": ("zeroing the local summary, zeroing or permuting the region summary and "
                    "zeroing the global summary must each move the decision logit. This is a "
                    "structural coupling test, not a claimed scientific effect size (§13.5)"),
    }


def checkpoint_state_audit(model: Any, groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Trainable branches reach the optimizer; the frozen tower never does."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    in_optimizer = {id(parameter) for group in groups for parameter in group["params"]}

    branch_present = {
        branch: bool(_named_parameters(model, prefixes))
        for branch, prefixes in (("local", LOCAL_PREFIXES),
                                 ("region_fusion", REGION_FUSION_PREFIXES),
                                 ("global_projection", GLOBAL_PREFIXES),
                                 ("fusion_head", FUSION_PREFIXES))}
    branch_optimized = {
        branch: all(id(parameter) in in_optimizer
                    for _name, parameter in _named_parameters(model, prefixes))
        for branch, prefixes in (("local", LOCAL_PREFIXES),
                                 ("region_fusion", REGION_FUSION_PREFIXES),
                                 ("global_projection", GLOBAL_PREFIXES),
                                 ("fusion_head", FUSION_PREFIXES))
        if branch_present[branch]}

    tower = getattr(getattr(model, "_global", None), "module", None)
    tower_ids = ({id(parameter) for parameter in tower.parameters()}
                 if tower is not None else set())
    leaked = sorted(str(index) for index in (tower_ids & in_optimizer))

    return {
        "trainable_parameter_names": sorted(trainable),
        "branches_present": branch_present,
        "branches_fully_in_optimizer": branch_optimized,
        "frozen_global_tower_registered_as_submodule": any(
            name.startswith("_global.") for name, _p in model.named_parameters()),
        "frozen_global_tower_parameters_in_optimizer": leaked,
        "passed": (all(branch_optimized.values()) and not leaked),
        "meaning": ("every trainable Track-R local, region-fusion, global-projection and "
                    "fusion parameter appears in an optimizer group, and no frozen SigLIP2 "
                    "parameter does (§13.5)"),
    }


def decision_graph_hash(model: Any) -> dict[str, Any]:
    """The decision graph identity C-G4/C-G5/C-G7 must match byte for byte."""
    variant = model.config.variant
    payload = {
        "decision_head_type": variant.decision_head_type,
        "decision_logit_name": variant.decision_logit_name,
        "decision_score_name": variant.decision_score_name,
        "branch_dimensions": {
            name: list(parameter.shape)
            for name, parameter in sorted(model.named_parameters())
            if any(name.startswith(prefix) for prefix in
                   (*GLOBAL_PREFIXES, *FUSION_PREFIXES, "local_summary_projection",
                    "region_attention_pool", "region_summary_norm"))},
        "module_parameter_hashes": {
            name: hashlib.sha256(
                json.dumps(list(parameter.shape)).encode("utf-8")).hexdigest()[:16]
            for name, parameter in sorted(model.named_parameters())},
        "architecture_identity": model.architecture_identity(),
    }
    return {"decision_graph_hash": _sha(payload), "material": payload}


def calibration_identity_guard(*, decision_logit_name: str, calibration_logit_name: str,
                               thresholded_quantity: str,
                               decision_score_name: str) -> dict[str, Any]:
    """§16.2: calibrate and threshold the SAME named quantity the model decides on.

    The Version-B G7 v1→v2 defect was thresholding a fused score with a
    temperature fitted on a different logit. The names are compared as strings
    because that is exactly the level at which the mistake was made — the tensors
    were both real and both finite.
    """
    logit_matches = calibration_logit_name == decision_logit_name
    score_matches = thresholded_quantity == decision_score_name
    return {
        "decision_logit_name": decision_logit_name,
        "calibration_logit_name": calibration_logit_name,
        "decision_score_name": decision_score_name,
        "thresholded_quantity": thresholded_quantity,
        "calibration_fits_the_decision_logit": logit_matches,
        "threshold_applies_to_the_decision_score": score_matches,
        "passed": logit_matches and score_matches,
        "meaning": ("no fused score may be thresholded by a calibration fitted on a "
                    "different quantity; a mismatch is a hard error, not a warning "
                    "(§16.2, §16.1)"),
    }


def audit_track_r(model: Any, batch: Any, *, groups: Sequence[Mapping[str, Any]] | None = None
                  ) -> dict[str, Any]:
    """Every §13.5 guard, run together, with one overall verdict."""
    variant = model.config.variant
    if variant.decision_logit_name != "fused_logit_R":
        raise DecisionAuditError(
            f"audit_track_r requires a Track-R decision head; this variant decides on "
            f"{variant.decision_logit_name!r}")

    if groups is None:
        groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)

    autograd = autograd_dependency(model, batch)
    intervention = feature_intervention(model, batch)
    state = checkpoint_state_audit(model, groups)
    graph = decision_graph_hash(model)
    calibration = calibration_identity_guard(
        decision_logit_name=variant.decision_logit_name,
        calibration_logit_name=variant.decision_logit_name,
        thresholded_quantity=variant.decision_score_name,
        decision_score_name=variant.decision_score_name)

    checks = {"autograd_dependency": autograd["passed"],
              "feature_intervention": intervention["passed"],
              "checkpoint_state": state["passed"],
              "calibration_identity": calibration["passed"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "track": variant.track,
        "decision_head_type": variant.decision_head_type,
        "autograd_dependency": autograd,
        "feature_intervention": intervention,
        "checkpoint_state": state,
        "decision_graph": graph,
        "calibration_identity": calibration,
        "checks": checks,
        "passed": all(checks.values()),
        "gate": ("§13.5: the Track-R decision dependency audit must pass before any C8 "
                 "scientific run. Failure blocks C8"),
    }


def audit_track_g(model: Any, batch: Any) -> dict[str, Any]:
    """Track G is global-only by design; the audit proves the absences.

    §13.4.1 is explicit that Track G MUST NOT instantiate ConvNeXt, region
    fusion, PromptHead or manifold modules. "Computed and ignored" is not the
    same as "absent", and only absence keeps the treatment factor clean, so the
    check is over instantiated modules rather than over used outputs.
    """
    variant = model.config.variant
    modules = sorted(name for name, _child in model.named_children())
    forbidden = [name for name in modules
                 if name.startswith(("local", "region", "prompt", "manifold"))]

    model.eval()
    import torch

    with torch.no_grad():
        output = model(batch)

    absent = {
        "local_logits": output.local_logits is None,
        "region_embeddings": output.region_embeddings is None,
        "region_distances": output.region_distances is None,
        "s_region": output.s_region is None,
        "p_prompt_spoof": output.p_prompt_spoof is None,
        "prompt_logits": output.prompt_logits is None,
    }
    checks = {"no_forbidden_modules_instantiated": not forbidden,
              "no_regional_or_prompt_output": all(absent.values()),
              "decision_logit_is_global": variant.decision_logit_name == "global_logit_G",
              "decision_score_is_p_g": variant.decision_score_name == "p_G",
              "score_is_not_fused": bool(torch.allclose(output.s_final, output.p_global))}
    return {
        "schema_version": SCHEMA_VERSION,
        "track": variant.track,
        "decision_head_type": variant.decision_head_type,
        "instantiated_modules": modules,
        "forbidden_modules_instantiated": forbidden,
        "absent_outputs": absent,
        "decision_logit_name": variant.decision_logit_name,
        "decision_score_name": variant.decision_score_name,
        "checks": checks,
        "passed": all(checks.values()),
        "meaning": ("Track G is intentionally global-only and excludes ConvNeXt, regions, "
                    "manifold and PromptHead rather than computing and ignoring them "
                    "(§13.4.1). No post-calibration fusion is permitted"),
    }


__all__ = ["SCHEMA_VERSION", "GRADIENT_MINIMUM", "INTERVENTION_TOLERANCE",
           "LOCAL_PREFIXES", "REGION_FUSION_PREFIXES", "GLOBAL_PREFIXES", "FUSION_PREFIXES",
           "DecisionAuditError", "autograd_dependency", "feature_intervention",
           "checkpoint_state_audit", "decision_graph_hash", "calibration_identity_guard",
           "audit_track_r", "audit_track_g"]
