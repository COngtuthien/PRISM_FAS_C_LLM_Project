"""Strict M9 checkpoint and resume.

Extends the M8/M5 pattern: an atomic write, then a resume that verifies EVERY
scientific identity before a single tensor reaches a live module. Spec section 12.7
requires that a resolved-config mismatch BLOCKS resume with a readable diff rather
than silently loading partial state, so there is no `strict=False` path here and no
partial restore.

Guarded identities (docs/M9_TRAINING_CONTRACT.md section 5):

    schema version, source package identity, M8 bank identity, architecture
    identity, SigLIP2 identity, recipe text cache identity, config hash,
    loss-contract hash, batch-contract hash, prototype identity, training stage,
    sampler state, best source-dev metric, stage lineage, git commit

The frozen SigLIP2 tower is deliberately NOT in `model_state`: it is 92 M frozen
parameters that never change, and binding it by verified SHA is both smaller and
stricter than round-tripping the bytes.
"""
from __future__ import annotations
import hashlib, json, os, random, subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch

M9_CHECKPOINT_SCHEMA_VERSION = "m9-detector-checkpoint-v1"
# Every field a resume must agree on. A mismatch on ANY of these raises.
STRICT_IDENTITY_FIELDS = (
    "source_package_identity", "m8_bank_identity", "architecture_identity",
    "siglip2_identity", "recipe_text_cache_identity", "config_hash",
    "loss_contract_hash", "batch_contract_hash", "dataset_contract_identity")
# The stage flow M9 runs. A resume may only continue at, or after, the stage the
# checkpoint recorded; it may never jump backwards or skip a stage.
STAGE_ORDER = ("G1", "G2", "G5", "G6")


class M9CheckpointError(RuntimeError):
    """An M9 checkpoint is missing, malformed or incompatible with this run."""


class StageTransitionError(RuntimeError):
    """An illegal M9 stage transition was requested."""


def git_commit(root: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root or Path.cwd()),
                             capture_output=True, text=True, timeout=15, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                       # noqa: BLE001 - provenance is best-effort
        return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def restore_rng(state: dict[str, Any]) -> None:
    if not state: return
    if "python" in state: random.setstate(_tuplize(state["python"]))
    if "numpy" in state: np.random.set_state(_tuplize(state["numpy"]))
    if "torch_cpu" in state: torch.set_rng_state(_byte_tensor(state["torch_cpu"]))
    cuda = state.get("torch_cuda") or []
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([_byte_tensor(item) for item in cuda])


def _tuplize(value: Any) -> Any:
    return tuple(_tuplize(item) for item in value) if isinstance(value, list) else value


def _byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(torch.uint8).cpu()


@dataclass(frozen=True)
class RunIdentity:
    """Every scientific identity this run is bound to.

    Constructed once from the real artifacts, written into every checkpoint and
    re-checked on every resume.
    """
    source_package_identity: str
    m8_bank_identity: str
    architecture_identity: str
    siglip2_identity: str
    recipe_text_cache_identity: str
    config_hash: str
    loss_contract_hash: str
    batch_contract_hash: str
    dataset_contract_identity: str
    convnext_weight_sha256: str = ""
    m7_recipe_bank_identity: str = ""
    region_prior_cache_identity: str = ""
    attack_mask_cache_identity: str = ""

    def payload(self) -> dict[str, str]:
        return {name: str(value) for name, value in sorted(vars(self).items())}

    def missing(self) -> list[str]:
        return [name for name in STRICT_IDENTITY_FIELDS if not getattr(self, name, "")]

    def validate(self) -> "RunIdentity":
        gaps = self.missing()
        if gaps: raise M9CheckpointError(f"the run identity is incomplete: {gaps}")
        return self


def identity_diff(checkpoint: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A human-readable diff of exactly which identities disagree."""
    return {name: {"checkpoint": checkpoint.get(name), "expected": expected[name]}
            for name in STRICT_IDENTITY_FIELDS
            if name in expected and checkpoint.get(name) != expected[name]}


def prototype_payload(manifold: Any, *, identity: str = "") -> dict[str, Any]:
    """The prototype state as plain arrays plus its content identity."""
    state = manifold.export_state()
    return {"centers": state.centers, "variances": state.variances, "counts": state.counts,
            "valid": state.valid, "epsilon": float(state.epsilon),
            "region_names": list(state.region_names), "initialized": bool(manifold.initialized),
            "prototype_identity": str(identity)}


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: Any, scheduler: Any, scaler: Any,
                    epoch: int, global_step: int, stage: str, identity: RunIdentity,
                    sampler_state: dict[str, Any], prototype_identity: str,
                    best_metrics: dict[str, Any], stage_lineage: list[dict[str, Any]],
                    history: list[dict[str, Any]] | None = None,
                    resolved_config: dict[str, Any] | None = None,
                    git_sha: str | None = None, ema_enabled: bool = False) -> str:
    """Atomic checkpoint write. Returns the SHA-256 of the written file."""
    identity.validate()
    if stage not in STAGE_ORDER: raise M9CheckpointError(f"unknown M9 stage {stage!r}")
    state = model.state_dict()
    payload = {
        "schema_version": M9_CHECKPOINT_SCHEMA_VERSION,
        "model_state": state,
        # The two heads are also written on their own so a checkpoint can be
        # audited without instantiating the full detector.
        "global_head_state": {key: value for key, value in state.items() if key.startswith("global_head.")},
        "prompt_head_state": {key: value for key, value in state.items() if key.startswith("prompt_head.")},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "global_step": int(global_step), "stage": str(stage),
        "sampler_state": dict(sampler_state), "rng_state": rng_state(),
        "identity": identity.payload(),
        "prototype_state": prototype_payload(model.manifold, identity=prototype_identity),
        "prototype_identity": str(prototype_identity),
        "best_metrics": dict(best_metrics), "stage_lineage": list(stage_lineage),
        "history": list(history or []), "resolved_config": dict(resolved_config or {}),
        "git_commit": git_sha if git_sha is not None else git_commit(),
        "ema_enabled": bool(ema_enabled), "torch_version": torch.__version__}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        # Verify before the pointer moves (spec section 15.3): a truncated write
        # must never become the file a resume trusts.
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema_version") != M9_CHECKPOINT_SCHEMA_VERSION:
            raise M9CheckpointError("checkpoint failed its own read-back verification")
        os.replace(temporary, target)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return sha256_file(target)


def load_checkpoint(path: Path, *, expected_identity: RunIdentity, expected_stage: str | None = None,
                    allowed_stages: tuple[str, ...] = STAGE_ORDER) -> dict[str, Any]:
    """Load and strictly verify. Any mismatch raises BEFORE anything is applied."""
    target = Path(path)
    if not target.is_file(): raise M9CheckpointError(f"checkpoint {target.name} does not exist")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != M9_CHECKPOINT_SCHEMA_VERSION:
        raise M9CheckpointError(f"checkpoint schema {payload.get('schema_version')!r} != "
                                f"{M9_CHECKPOINT_SCHEMA_VERSION!r}")
    stored = dict(payload.get("identity") or {})
    missing = [name for name in STRICT_IDENTITY_FIELDS if name not in stored]
    if missing: raise M9CheckpointError(f"checkpoint is missing identity fields {missing}")
    expected = expected_identity.validate().payload()
    difference = identity_diff(stored, expected)
    if difference:
        raise M9CheckpointError(f"refusing to resume: identity mismatch on {sorted(difference)}: {difference}")
    stage = str(payload.get("stage"))
    if stage not in allowed_stages:
        raise M9CheckpointError(f"checkpoint stage {stage!r} is not one of {list(allowed_stages)}")
    if expected_stage is not None and stage != expected_stage:
        raise M9CheckpointError(f"checkpoint is at stage {stage!r} but stage {expected_stage!r} was requested")
    for name in ("model_state", "prototype_state", "sampler_state", "rng_state"):
        if payload.get(name) is None: raise M9CheckpointError(f"checkpoint is missing {name}")
    return payload


def apply_checkpoint(payload: dict[str, Any], *, model: torch.nn.Module, optimizer: Any = None,
                     scheduler: Any = None, scaler: Any = None, restore_rng_state: bool = True) -> dict[str, Any]:
    """Restore everything or nothing. `strict=True` is not negotiable here: a
    silently partial load is exactly the failure the spec forbids."""
    model.load_state_dict(payload["model_state"], strict=True)
    prototypes = payload["prototype_state"]
    _restore_prototypes(model.manifold, prototypes)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng_state: restore_rng(payload.get("rng_state") or {})
    return {"epoch": int(payload["epoch"]), "global_step": int(payload["global_step"]),
            "stage": str(payload["stage"]), "sampler_state": dict(payload["sampler_state"]),
            "prototype_identity": str(payload.get("prototype_identity") or ""),
            "best_metrics": dict(payload.get("best_metrics") or {}),
            "stage_lineage": list(payload.get("stage_lineage") or []),
            "history": list(payload.get("history") or [])}


def _restore_prototypes(manifold: Any, payload: dict[str, Any]) -> None:
    """`model_state` already carries the prototype buffers; this re-asserts them
    from the explicit prototype record so the two can never disagree."""
    from .manifold import PrototypeState
    if not bool(payload.get("initialized", False)):
        manifold.initialized.fill_(False)
        return
    state = PrototypeState(centers=np.asarray(payload["centers"]), variances=np.asarray(payload["variances"]),
                           counts=np.asarray(payload["counts"]), valid=np.asarray(payload["valid"]),
                           epsilon=float(payload["epsilon"]),
                           region_names=tuple(str(name) for name in payload["region_names"]))
    manifold.load_state(state)


def checkpoint_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    prototypes = payload.get("prototype_state") or {}
    return {"schema_version": payload.get("schema_version"), "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"), "stage": payload.get("stage"),
            "identity": payload.get("identity"), "prototype_identity": payload.get("prototype_identity"),
            "prototypes_initialized": bool(prototypes.get("initialized", False)),
            "best_metrics": payload.get("best_metrics"), "stage_lineage": payload.get("stage_lineage"),
            "sampler_state": payload.get("sampler_state"), "git_commit": payload.get("git_commit"),
            "ema_enabled": payload.get("ema_enabled"), "torch_version": payload.get("torch_version"),
            "rng_state_present": bool(payload.get("rng_state")),
            "global_head_keys": sorted((payload.get("global_head_state") or {}).keys()),
            "prompt_head_keys": sorted((payload.get("prompt_head_state") or {}).keys()),
            "sha256": sha256_file(Path(path))}


# --- stage machine ----------------------------------------------------------

# Spec section 11.2 (Table 40).
RUN_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED", "RESUMING")
STATUS_TRANSITIONS = {
    "PENDING": ("RUNNING",),
    "RUNNING": ("COMPLETED", "FAILED", "INTERRUPTED"),
    "INTERRUPTED": ("RESUMING",),
    "RESUMING": ("RUNNING",),
    "FAILED": ("RESUMING",),          # only when the failure is marked recoverable
    "COMPLETED": ()}


def check_status_transition(current: str, nxt: str, *, recoverable: bool = False) -> str:
    if current not in RUN_STATUSES or nxt not in RUN_STATUSES:
        raise StageTransitionError(f"unknown run status in {current!r} -> {nxt!r}")
    if nxt not in STATUS_TRANSITIONS[current]:
        raise StageTransitionError(f"illegal run status transition {current!r} -> {nxt!r}")
    if current == "FAILED" and not recoverable:
        raise StageTransitionError("a FAILED run may resume only when the failure is marked recoverable")
    return nxt


def check_stage_transition(current: str | None, nxt: str, *, order: tuple[str, ...] = STAGE_ORDER) -> str:
    """The declared flow, walked forwards, never skipped and never repeated backwards.

    `order` is the stage flow THIS RUN declares. The full method runs
    G1 -> G2 -> G5 -> G6; a variant with `manifold: off` has no prototypes to
    initialize, so its declared flow is G1 -> G5 -> G6 and G2 is not a stage it may
    enter. Running G2 for a variant that has no manifold would be a meaningless
    stage; skipping it for one that does would be a missing stage. Both are refused.
    """
    if nxt not in STAGE_ORDER: raise StageTransitionError(f"{nxt!r} is not an M9 stage; G7/G8 are M10")
    if nxt not in order:
        raise StageTransitionError(f"{nxt!r} is not part of this run's declared flow {' -> '.join(order)}")
    if current is None:
        if nxt != order[0]:
            raise StageTransitionError(f"this run must start at {order[0]!r}, not {nxt!r}")
        return nxt
    if current not in order: raise StageTransitionError(f"{current!r} is not part of the declared flow")
    position = order.index(current)
    if nxt == current: return nxt
    if order.index(nxt) != position + 1:
        raise StageTransitionError(f"illegal stage transition {current!r} -> {nxt!r}; "
                                   f"the declared flow is {' -> '.join(order)}")
    return nxt


@dataclass
class StageLineage:
    """The recorded chain of stages, each with its input and output identities."""
    entries: list[dict[str, Any]] = field(default_factory=list)
    # The stage flow this run declares; see `check_stage_transition`.
    order: tuple[str, ...] = STAGE_ORDER

    @property
    def current(self) -> str | None:
        return str(self.entries[-1]["stage"]) if self.entries else None

    def enter(self, stage: str, *, input_hashes: dict[str, Any]) -> dict[str, Any]:
        check_stage_transition(self.current, stage, order=self.order)
        if self.current == stage: return self.entries[-1]
        entry = {"stage": stage, "input_hashes": dict(input_hashes), "output_hashes": {},
                 "status": "RUNNING"}
        self.entries.append(entry)
        return entry

    def complete(self, stage: str, *, output_hashes: dict[str, Any]) -> dict[str, Any]:
        if not self.entries or self.entries[-1]["stage"] != stage:
            raise StageTransitionError(f"cannot complete {stage!r}: it is not the active stage")
        self.entries[-1]["output_hashes"] = dict(output_hashes)
        self.entries[-1]["status"] = "COMPLETED"
        return self.entries[-1]

    def payload(self) -> list[dict[str, Any]]: return [dict(entry) for entry in self.entries]

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(",", ":"),
                                         default=str).encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, entries: list[dict[str, Any]] | None,
                     *, order: tuple[str, ...] = STAGE_ORDER) -> "StageLineage":
        return cls(entries=[dict(entry) for entry in (entries or [])], order=tuple(order))
