"""A09 — bounded PC-vs-Modal backend parity for the B08 configuration.

Table 60 asks "PC vs Modal same seed/config". A full-length 35-epoch B08 run on the
PC backend is BLOCKED (this host is `torch 2.13.0+cpu`, CUDA False) and that row
stays in the matrix with its reason. What runs instead is the M6 bounded step-parity
protocol, reused verbatim: the same scientific configuration and the same seed, a
declared handful of optimizer steps on each backend, compared field by field
against tolerances that were fixed in `configs/cloud/modal_m6.yaml` before any
result existed.

**This is parity evidence, not a superiority experiment.** It answers "does the
infrastructure move the numbers", and it is reported as a parity result. It is
never a second full training result, it selects no checkpoint, and it produces no
target prediction.

What a probe records per step, so the comparison is field-by-field rather than a
single scalar:

```
sample ids      the exact batch, in order            compared EXACTLY
labels          the exact label vector               compared EXACTLY
composition     real live / real spoof / synthetic   compared EXACTLY
loss terms      every declared term, independently   compared to loss_abs_diff
global logits   the raw forward output               compared to logit_* tolerances
p_global        the probability                      compared to probability_max_abs_diff
```

An exact-match failure on the batch is a different finding from a numeric drift on
the losses, and collapsing them would hide which one happened. The batch is a pure
function of `(seed, epoch, pool identities)`, so it MUST match exactly on both
backends; the numerics may differ within tolerance because one side runs fp32 on a
CPU and the other may use AMP on an L4.

Never imports modal.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

BACKEND_PARITY_SCHEMA_VERSION = "m10-backend-parity-v1"
PARITY_PROTOCOL = "bounded_step_parity"
# The M6 contract, reused verbatim. Pre-declared; never widened after seeing a result.
M6_TOLERANCES = {
    "logit_max_abs_diff": 1.0e-3,
    "logit_mean_abs_diff": 2.0e-4,
    "probability_max_abs_diff": 2.5e-4,
    "loss_abs_diff": 5.0e-4,
    "feature_min_cosine": 0.99999,
}
# Fields of a step record that must agree EXACTLY on both backends.
EXACT_FIELDS = ("step", "sample_ids", "labels", "composition", "stage")


class BackendParityError(RuntimeError):
    """A bounded parity probe cannot be built or compared as declared."""


def load_shared_weights(trainer: Any, checkpoint: Path) -> dict[str, Any]:
    """Start BOTH backends from the same weights.

    Without this the probe compares two independently initialized models. The
    trainer seeds before building, but the dataset is constructed in between, and it
    does different work on the two hosts — building a region-prior cache on one and
    loading it on the other — so the RNG stream reaching the head initialization
    diverges and the two backends legitimately hold different weights. Measured: a
    1.11 max absolute logit difference against a 1e-3 tolerance, identical whether
    the remote side ran bf16 or fp32, which is what proved it was not a precision
    effect.

    The M6 parity contract already anticipated this — it drives both sides from a
    shared `parity_inputs` directory. This is the same idea: load one frozen
    checkpoint on both backends, then compare the compute that follows.
    """
    import torch
    from prism_fas.detector.checkpoint import _restore_prototypes, sha256_file
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    _restore_prototypes(trainer.model.manifold, payload["prototype_state"])
    return {"checkpoint_sha256": sha256_file(Path(checkpoint)),
            "stage": str(payload.get("stage")), "global_step": int(payload.get("global_step", 0)),
            "architecture_identity": (payload.get("identity") or {}).get("architecture_identity")}


def probe_steps(trainer: Any, *, steps: int, stage: str = "G1") -> list[dict[str, Any]]:
    """Run `steps` optimizer steps and record what the comparison needs.

    Uses the SAME `M9Trainer` both backends use — the probe is not a second training
    path, it is the real one, stopped early.
    """
    import torch
    records: list[dict[str, Any]] = []
    sampler = trainer.samplers[stage]
    trainer.step_in_epoch = 0
    for plan in sampler.iter_epoch(0, start_step=0):
        if len(records) >= int(steps): break
        batch = trainer.dataset.batch_from_plan(plan)
        with torch.no_grad():
            moved = batch.to(trainer.device)
            output = trainer.model(moved)
            logits = output.global_logit.detach().float().cpu().reshape(-1)
            probabilities = output.p_global.detach().float().cpu().reshape(-1)
        result = trainer.train_step(plan, stage)
        from prism_fas.detector.dataset import batch_composition
        records.append({
            "step": int(plan.step), "stage": stage,
            "sample_ids": list(batch.sample_ids),
            "labels": [int(value) for value in batch.label.tolist()],
            "composition": batch_composition(batch),
            "global_logits": [float(value) for value in logits.tolist()],
            "p_global": [float(value) for value in probabilities.tolist()],
            "loss_terms": {name: float(result[name]) for name in sorted(result)
                           if name.startswith("L_")},
            "grad_norm": float(result["grad_norm"])})
    if len(records) != int(steps):
        raise BackendParityError(f"probe produced {len(records)} steps, expected {steps}")
    return records


def probe_payload(trainer: Any, *, backend: str, steps: int, stage: str = "G1",
                  device_report: dict[str, Any] | None = None,
                  shared_checkpoint: Path | None = None) -> dict[str, Any]:
    """One backend's half of the parity evidence, with the identities it binds."""
    shared = (load_shared_weights(trainer, Path(shared_checkpoint))
              if shared_checkpoint is not None else None)
    records = probe_steps(trainer, steps=steps, stage=stage)
    return {"shared_checkpoint": shared,
            "schema_version": BACKEND_PARITY_SCHEMA_VERSION, "protocol": PARITY_PROTOCOL,
            "backend": str(backend), "steps": int(steps), "stage": stage,
            "device": trainer.device, "device_report": dict(device_report or {}),
            "seed": int(trainer.config.seed),
            "amp": {"enabled": trainer.amp_enabled, "dtype": str(trainer.amp_dtype)},
            "identity": trainer.identity.payload(),
            "variant_identity": trainer.variant.identity(),
            "batch_contract": trainer.samplers[stage].contract.payload(),
            "sampler_fingerprint": trainer.samplers[stage].fingerprint(0),
            "parameter_counts": trainer.model.parameter_counts(),
            "records": records,
            "target_test_opened": False, "target_labels_opened": False}


def _exact_mismatches(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(zip(left, right)):
        for field in EXACT_FIELDS:
            if a.get(field) != b.get(field):
                problems.append({"step_index": index, "field": field,
                                 "local": a.get(field), "remote": b.get(field)})
    return problems


def _numeric(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str,
             *, max_abs: float, mean_abs: float | None = None) -> dict[str, Any]:
    import numpy as np
    a = np.asarray([value for record in left for value in record[key]], dtype=np.float64)
    b = np.asarray([value for record in right for value in record[key]], dtype=np.float64)
    if a.shape != b.shape:
        return {"passed": False, "reason": f"{key} shapes differ: {a.shape} vs {b.shape}"}
    difference = np.abs(a - b)
    observed_max = float(difference.max()) if difference.size else 0.0
    observed_mean = float(difference.mean()) if difference.size else 0.0
    passed = observed_max <= max_abs and (mean_abs is None or observed_mean <= mean_abs)
    return {"passed": bool(passed), "compared": int(a.size),
            "max_abs_diff": observed_max, "mean_abs_diff": observed_mean,
            "tolerance_max": max_abs, "tolerance_mean": mean_abs}


def compare_backends(local: dict[str, Any], remote: dict[str, Any], *,
                     tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare the two halves. Reports; raises nothing.

    A parity result that quietly widened its own tolerance would be worthless, so
    the tolerances come in from the frozen M6 block and are echoed into the report
    beside every observed value.
    """
    limits = {**M6_TOLERANCES, **(tolerances or {})}
    left, right = local["records"], remote["records"]
    checks: dict[str, Any] = {}

    # The scientific configuration must be the SAME on both sides; that is what
    # makes this a backend comparison rather than two different experiments.
    shared_identity = {name: {"local": local["identity"].get(name), "remote": remote["identity"].get(name)}
                       for name in sorted(set(local["identity"]) | set(remote["identity"]))
                       if local["identity"].get(name) != remote["identity"].get(name)}
    checks["scientific_identity_matches"] = {"passed": not shared_identity,
                                             "differing": shared_identity}
    checks["variant_identity_matches"] = {
        "passed": local["variant_identity"] == remote["variant_identity"],
        "local": local["variant_identity"], "remote": remote["variant_identity"]}
    checks["seed_matches"] = {"passed": local["seed"] == remote["seed"],
                              "local": local["seed"], "remote": remote["seed"]}
    # Both halves must start from the SAME weights, or this measures initialization
    # rather than the backend.
    local_shared, remote_shared = local.get("shared_checkpoint"), remote.get("shared_checkpoint")
    checks["started_from_the_same_weights"] = {
        "passed": bool(local_shared) and bool(remote_shared)
                  and local_shared["checkpoint_sha256"] == remote_shared["checkpoint_sha256"],
        "local": (local_shared or {}).get("checkpoint_sha256"),
        "remote": (remote_shared or {}).get("checkpoint_sha256")}
    checks["step_count_matches"] = {"passed": len(left) == len(right),
                                    "local": len(left), "remote": len(right)}
    # The batch is a pure function of (seed, epoch, pool identities), so it must be
    # EXACT. A drift here is a determinism failure, not a numeric one.
    exact = _exact_mismatches(left, right) if len(left) == len(right) else [{"field": "step_count"}]
    checks["batch_identity_exact"] = {"passed": not exact, "mismatches": exact[:10],
                                      "mismatch_count": len(exact)}
    checks["sampler_fingerprint_matches"] = {
        "passed": local["sampler_fingerprint"] == remote["sampler_fingerprint"],
        "local": local["sampler_fingerprint"], "remote": remote["sampler_fingerprint"]}

    if len(left) == len(right) and left:
        checks["global_logits"] = _numeric(left, right, "global_logits",
                                           max_abs=limits["logit_max_abs_diff"],
                                           mean_abs=limits["logit_mean_abs_diff"])
        checks["p_global"] = _numeric(left, right, "p_global",
                                      max_abs=limits["probability_max_abs_diff"])
        terms = sorted({name for record in left for name in record["loss_terms"]})
        loss_checks: dict[str, Any] = {}
        for name in terms:
            a = [record["loss_terms"].get(name, 0.0) for record in left]
            b = [record["loss_terms"].get(name, 0.0) for record in right]
            worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
            loss_checks[name] = {"passed": worst <= limits["loss_abs_diff"],
                                 "max_abs_diff": float(worst),
                                 "tolerance": limits["loss_abs_diff"]}
        checks["loss_trajectory"] = {"passed": all(item["passed"] for item in loss_checks.values()),
                                     "per_term": loss_checks}

    passed = all(check.get("passed") for check in checks.values())
    body = {"schema_version": BACKEND_PARITY_SCHEMA_VERSION, "protocol": PARITY_PROTOCOL,
            "kind": "parity_not_superiority",
            "passed": bool(passed), "checks": checks, "tolerances": limits,
            "local_backend": {"backend": local["backend"], "device": local["device"],
                              "amp": local["amp"], "device_report": local.get("device_report")},
            "remote_backend": {"backend": remote["backend"], "device": remote["device"],
                               "amp": remote["amp"], "device_report": remote.get("device_report")},
            "steps": local["steps"], "stage": local["stage"],
            "target_test_opened": False, "target_labels_opened": False,
            "note": ("Bounded step parity, the M6 protocol reused verbatim. It answers whether the "
                     "infrastructure moves the numbers at identical configuration and seed. It is "
                     "never reported as a second full training result and carries no superiority "
                     "claim; the full-length PC training row stays BLOCKED for its declared reason.")}
    return {**body, "parity_identity": hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()}


def write_parity(path: Path, payload: dict[str, Any]) -> Path:
    from prism_fas.utils.core import atomic_json_write
    atomic_json_write(Path(path), payload)
    return Path(path)
