from __future__ import annotations
import hashlib, os, random, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION

# Every identity a resume must agree on. A mismatch raises; nothing is ever
# partially loaded.
STRICT_IDENTITY_FIELDS = ("package_identity", "recipe_bank_identity", "pair_plan_identity",
                          "config_hash", "architecture_hash", "adaface_weight_sha256")


class CheckpointError(RuntimeError):
    """A GPAT checkpoint is missing, malformed or incompatible with this run."""


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, scaler: Any, epoch: int, global_step: int,
                    best_metrics: dict[str, Any], identity: dict[str, str], history: list[dict[str, Any]],
                    record_set_hashes: dict[str, str], git_commit: str | None,
                    resume_lineage: list[dict[str, Any]] | None = None) -> str:
    """Atomic checkpoint write. Returns the SHA-256 of the written file."""
    payload = {
        "schema_version": GPAT_CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "global_step": int(global_step),
        "best_metrics": dict(best_metrics), "history": list(history),
        "rng_state": rng_state(), "identity": dict(identity),
        "record_set_hashes": dict(record_set_hashes), "git_commit": git_commit,
        "resume_lineage": list(resume_lineage or []),
        "torch_version": torch.__version__}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return sha256_file(path)


def load_checkpoint(path: Path, *, expected_identity: dict[str, str]) -> dict[str, Any]:
    """Load and strictly verify. Any identity mismatch raises before a single
    tensor is copied into a live module."""
    path = Path(path)
    if not path.is_file(): raise CheckpointError(f"checkpoint {path.name} does not exist")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != GPAT_CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"checkpoint schema {payload.get('schema_version')!r} != {GPAT_CHECKPOINT_SCHEMA_VERSION!r}")
    identity = payload.get("identity") or {}
    mismatched = [field for field in STRICT_IDENTITY_FIELDS
                  if field in expected_identity and identity.get(field) != expected_identity[field]]
    if mismatched:
        details = {field: {"checkpoint": identity.get(field), "expected": expected_identity[field]} for field in mismatched}
        raise CheckpointError(f"refusing to resume: identity mismatch on {mismatched}: {details}")
    missing = [field for field in STRICT_IDENTITY_FIELDS if field not in identity]
    if missing: raise CheckpointError(f"checkpoint is missing identity fields {missing}")
    return payload


def apply_checkpoint(payload: dict[str, Any], *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                     scheduler: Any = None, scaler: Any = None, restore_rng_state: bool = True) -> dict[str, Any]:
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng_state: restore_rng(payload.get("rng_state") or {})
    return {"epoch": int(payload["epoch"]), "global_step": int(payload["global_step"]),
            "best_metrics": dict(payload.get("best_metrics") or {}), "history": list(payload.get("history") or []),
            "resume_lineage": list(payload.get("resume_lineage") or [])}


def checkpoint_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {"schema_version": payload.get("schema_version"), "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"), "best_metrics": payload.get("best_metrics"),
            "identity": payload.get("identity"), "sha256": sha256_file(Path(path)),
            "git_commit": payload.get("git_commit"), "torch_version": payload.get("torch_version"),
            "rng_state_present": bool(payload.get("rng_state")),
            "resume_lineage": payload.get("resume_lineage")}
