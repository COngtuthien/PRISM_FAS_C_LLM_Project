from __future__ import annotations
import hashlib, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from prism_fas.utils.core import atomic_json_write, git_commit
from .gpat_checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint, sha256_file
from .gpat_contracts import LL_INVARIANT_TOLERANCE
from .gpat_losses import assert_invariants, compute_losses, loss_manifest
from .gpat_model import build_gpat_model
from .m8_pipeline import SampleStore, SourceOnlyAudit, build_batch, config_hash, load_pairs, resolve_bank

TRAINER_SCHEMA_VERSION = "m8-gpat-trainer-v1"


class TrainerError(RuntimeError):
    """GPAT training could not proceed under the declared contract."""


def seed_everything(seed: int) -> None:
    """Seed only Torch, which is what model initialization actually consumes.

    The NumPy and Python global generators are deliberately left alone: every
    M8 sampling decision (batch order, operator seeds, mask coverage) comes from
    an explicitly seeded local PCG64, so seeding the globals would suggest a
    dependency that must not exist. The M7 guard asserts this for the whole
    synthesis package.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str | None) -> str:
    if requested: return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_schedule(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_fraction: float,
                    min_lr: float) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(round(total_steps * float(warmup_fraction))))
    base = [group["lr"] for group in optimizer.param_groups]

    def factor(step: int, index: int) -> float:
        peak = base[index] if base[index] > 0 else 1.0
        if step < warmup: return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (float(min_lr) + (peak - float(min_lr)) * cosine) / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda step, index=index: factor(step, index)
                                                         for index in range(len(base))])


def batch_slices(count: int, batch_size: int, *, seed: int, epoch: int, shuffle: bool) -> list[list[int]]:
    """Deterministic index batches: a local RNG seeded by (seed, epoch), never
    the global RNG."""
    order = list(range(count))
    if shuffle:
        # Python's hash() is randomized per process for str keys, so it must
        # never seed anything that has to be reproducible across runs.
        material = f"gpat|{int(seed)}|{int(epoch)}".encode("utf-8")
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2 ** 32)
        generator = np.random.Generator(np.random.PCG64(derived))
        order = [order[index] for index in generator.permutation(count)]
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


@dataclass
class GPATTrainer:
    """Device-neutral GPAT trainer. Never imports modal.

    The Modal wrapper only chooses the device and the run root; this class runs
    identically on a CPU smoke and on an L4.
    """
    config: dict[str, Any]
    package_root: Path
    bank_root: Path
    pairs_root: Path
    run_root: Path
    weight_root: Path
    device: str = "cpu"
    identity_model: Any = None

    def __post_init__(self) -> None:
        self.package_root = Path(self.package_root); self.bank_root = Path(self.bank_root)
        self.pairs_root = Path(self.pairs_root); self.run_root = Path(self.run_root)
        self.audit = SourceOnlyAudit()
        self.store = SampleStore.open(self.package_root, self.audit)
        self.bank = resolve_bank(self.bank_root)
        self.train_pairs = load_pairs(self.pairs_root, "train")
        self.validation_pairs = load_pairs(self.pairs_root, "validation")
        self.package_identity = json.loads((self.package_root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"]
        self.bank_identity = self.bank["lock"]["bank_content_identity_sha256"]
        self.pair_plan_identity = json.loads((self.pairs_root / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))["pair_plan_identity_sha256"]
        self.config_hash = config_hash(self.config)
        self.model = build_gpat_model(self.config).to(self.device)
        self.architecture_hash = self.model.architecture_hash()
        if self.identity_model is None:
            from .quality_models import QualityModelRegistry
            self.registry = QualityModelRegistry.resolve(self.weight_root, roles=("identity",))
            self.identity_model = self.registry.adaface(self.device)
        else:
            from .quality_models import QualityModelRegistry
            self.registry = QualityModelRegistry.resolve(self.weight_root, roles=("identity",), verify=False)
        self.adaface_sha = self.registry.verified["identity"]
        self.use_amp = bool(self.device.startswith("cuda") and str(self.config["precision"]["cuda"]).lower() == "fp16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(self.config),
            weight_decay=float(self.config["optimizer"]["weight_decay"]),
            betas=tuple(float(value) for value in self.config["optimizer"]["betas"]))
        self.history: list[dict[str, Any]] = []
        self.resume_lineage: list[dict[str, Any]] = []

    # --- identity -----------------------------------------------------------
    def identity(self) -> dict[str, str]:
        return {"package_identity": self.package_identity, "recipe_bank_identity": self.bank_identity,
                "pair_plan_identity": self.pair_plan_identity, "config_hash": self.config_hash,
                "architecture_hash": self.architecture_hash, "adaface_weight_sha256": self.adaface_sha}

    def record_set_hashes(self) -> dict[str, str]:
        import hashlib
        out = {}
        for name, rows in (("train", self.train_pairs), ("validation", self.validation_pairs)):
            for role in ("live_source_record_id", "spoof_source_record_id"):
                material = "|".join(sorted({row[role] for row in rows}))
                out[f"{name}_{role}"] = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return out

    # --- steps --------------------------------------------------------------
    def _batch(self, pairs: list[dict[str, Any]]):
        return build_batch(self.store, pairs, self.bank, self.identity_model, device=self.device)

    def _forward(self, batch):
        autocast = torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp)
        with autocast:
            output = self.model.forward_batch(batch)
            embedding = self.identity_model(output.synthetic_image)
        result = compute_losses(output, batch, embedding.float(), self.config.get("loss"))
        assert_invariants(result, ll_tolerance=float(self.config["invariants"]["ll_max_abs_error"]),
                          outside_tolerance=float(self.config["invariants"]["outside_mask_max_abs_error"]))
        return output, result

    def train_step(self, pairs: list[dict[str, Any]], scheduler: Any) -> dict[str, Any]:
        self.model.train()
        batch = self._batch(pairs)
        self.optimizer.zero_grad(set_to_none=True)
        _, result = self._forward(batch)
        self.scaler.scale(result.total).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                         float(self.config["gradient_clip_norm"])).item())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if scheduler is not None: scheduler.step()
        metrics = result.detached()
        metrics.update({"grad_norm": grad_norm, "batch_size": batch.batch_size,
                        "lr": float(self.optimizer.param_groups[-1]["lr"])})
        return metrics

    @torch.no_grad()
    def validate(self, *, limit_batches: int | None = None) -> dict[str, Any]:
        self.model.eval()
        batches = batch_slices(len(self.validation_pairs), int(self.config["batch_size"]),
                               seed=int(self.config["seed"]), epoch=0, shuffle=False)
        if limit_batches is not None: batches = batches[:limit_batches]
        totals: dict[str, float] = {}
        count = 0
        for indices in batches:
            batch = self._batch([self.validation_pairs[index] for index in indices])
            _, result = self._forward(batch)
            for key, value in result.detached().items(): totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
        if not count: raise TrainerError("validation produced no batches")
        # `total` is exposed as `validation_total_loss` so it matches the
        # checkpoint_selection.primary key declared in the config and contract.
        return {f"validation_{'total_loss' if key == 'total' else key}": value / count
                for key, value in totals.items()}

    # --- loops --------------------------------------------------------------
    def smoke(self, *, steps: int = 5, resume_steps: int = 6, run_id: str = "gpat_m8_smoke") -> dict[str, Any]:
        """5 optimizer steps, checkpoint, then resume and continue past `steps`."""
        seed_everything(int(self.config["seed"]))
        batches = batch_slices(len(self.train_pairs), int(self.config["batch_size"]),
                               seed=int(self.config["seed"]), epoch=0, shuffle=True)
        scheduler = cosine_schedule(self.optimizer, total_steps=max(resume_steps, steps),
                                    warmup_fraction=float(self.config["scheduler"]["warmup_fraction"]),
                                    min_lr=float(self.config["scheduler"]["min_lr"]))
        first: list[dict[str, Any]] = []
        for step in range(steps):
            first.append(self.train_step([self.train_pairs[index] for index in batches[step]], scheduler))
        checkpoint_path = self.run_root / "checkpoints" / "last.pt"
        digest = save_checkpoint(checkpoint_path, model=self.model, optimizer=self.optimizer, scheduler=scheduler,
                                 scaler=self.scaler, epoch=0, global_step=steps, best_metrics={},
                                 identity=self.identity(), history=first,
                                 record_set_hashes=self.record_set_hashes(), git_commit=git_commit(Path.cwd()))
        payload = load_checkpoint(checkpoint_path, expected_identity=self.identity())
        state = apply_checkpoint(payload, model=self.model, optimizer=self.optimizer, scheduler=scheduler,
                                 scaler=self.scaler)
        resumed: list[dict[str, Any]] = []
        for step in range(state["global_step"], resume_steps):
            resumed.append(self.train_step([self.train_pairs[index] for index in batches[step]], scheduler))
        final = save_checkpoint(checkpoint_path, model=self.model, optimizer=self.optimizer, scheduler=scheduler,
                                scaler=self.scaler, epoch=0, global_step=resume_steps, best_metrics={},
                                identity=self.identity(), history=first + resumed,
                                record_set_hashes=self.record_set_hashes(), git_commit=git_commit(Path.cwd()),
                                resume_lineage=[{"resumed_from_step": state["global_step"], "checkpoint_sha256": digest}])
        return {"run_id": run_id, "steps_first": steps, "steps_after_resume": resume_steps,
                "resume_continued": bool(state["global_step"] == steps and resumed),
                "resumed_from_step": state["global_step"],
                "first_metrics": first, "resumed_metrics": resumed,
                "losses": [entry["total"] for entry in first + resumed],
                "losses_finite": all(math.isfinite(entry["total"]) for entry in first + resumed),
                "ll_invariant_max": max(entry["ll_invariant_max_abs_error"] for entry in first + resumed),
                "outside_mask_max": max(entry["outside_mask_max_abs_error"] for entry in first + resumed),
                "checkpoint_sha256_first": digest, "checkpoint_sha256_final": final,
                "amp": self.use_amp, "device": self.device, "identity": self.identity(),
                "source_isolation": self.audit.report()}

    def fit(self, *, run_id: str, progress: Callable[[dict[str, Any]], None] | None = None,
            max_epochs: int | None = None, limit_steps_per_epoch: int | None = None,
            resume: bool = False) -> dict[str, Any]:
        seed_everything(int(self.config["seed"]))
        epochs = int(max_epochs if max_epochs is not None else self.config["epochs"])
        batch_size = int(self.config["batch_size"])
        steps_per_epoch = len(batch_slices(len(self.train_pairs), batch_size, seed=0, epoch=0, shuffle=False))
        if limit_steps_per_epoch is not None: steps_per_epoch = min(steps_per_epoch, limit_steps_per_epoch)
        scheduler = cosine_schedule(self.optimizer, total_steps=max(1, epochs * steps_per_epoch),
                                    warmup_fraction=float(self.config["scheduler"]["warmup_fraction"]),
                                    min_lr=float(self.config["scheduler"]["min_lr"]))
        checkpoints = self.run_root / "checkpoints"
        last_path, best_path = checkpoints / "last.pt", checkpoints / "best.pt"
        start_epoch, global_step = 0, 0
        best = {"validation_total_loss": float("inf"), "validation_identity_cosine": -1.0, "epoch": -1}
        if resume and last_path.is_file():
            payload = load_checkpoint(last_path, expected_identity=self.identity())
            state = apply_checkpoint(payload, model=self.model, optimizer=self.optimizer, scheduler=scheduler,
                                     scaler=self.scaler)
            start_epoch, global_step = int(state["epoch"]) + 1, int(state["global_step"])
            best = {**best, **state["best_metrics"]}
            self.history = state["history"]
            self.resume_lineage = state["resume_lineage"] + [{"resumed_at_epoch": start_epoch, "global_step": global_step}]
        selection = self.config["checkpoint_selection"]
        stopping = self.config["early_stopping"]
        stop_reason, epochs_without_improvement = "completed_all_epochs", 0
        started = time.monotonic()
        for epoch in range(start_epoch, epochs):
            batches = batch_slices(len(self.train_pairs), batch_size, seed=int(self.config["seed"]),
                                   epoch=epoch, shuffle=True)[:steps_per_epoch]
            epoch_metrics: list[dict[str, Any]] = []
            for position, indices in enumerate(batches, 1):
                metrics = self.train_step([self.train_pairs[index] for index in indices], scheduler)
                global_step += 1
                epoch_metrics.append(metrics)
                if progress and (position == 1 or position % 10 == 0 or position == len(batches)):
                    progress({"stage": "train", "epoch": epoch, "step": position, "steps": len(batches),
                              "global_step": global_step, "total": round(metrics["total"], 5)})
            validation = self.validate()
            summary = {"epoch": epoch, "global_step": global_step,
                       "train_total": float(np.mean([entry["total"] for entry in epoch_metrics])),
                       "train_grad_norm": float(np.mean([entry["grad_norm"] for entry in epoch_metrics])),
                       "elapsed_seconds": round(time.monotonic() - started, 1), **validation}
            for key in ("style", "identity", "map", "strength", "total_variation", "residual"):
                summary[f"train_{key}"] = float(np.mean([entry[key] for entry in epoch_metrics]))
            if not math.isfinite(summary["train_total"]): raise TrainerError(f"epoch {epoch} produced a non-finite training loss")
            self.history.append(summary)
            improved = summary[selection["primary"]] < best[selection["primary"]] - 1e-9
            tied = abs(summary[selection["primary"]] - best[selection["primary"]]) <= 1e-9
            if tied and summary.get(selection["tie_breaker"], -1.0) > best.get(selection["tie_breaker"], -1.0):
                improved = True
            if improved:
                best = {selection["primary"]: summary[selection["primary"]],
                        selection["tie_breaker"]: summary.get(selection["tie_breaker"], 0.0),
                        "epoch": epoch, "global_step": global_step}
                best["sha256"] = save_checkpoint(best_path, model=self.model, optimizer=self.optimizer,
                                                 scheduler=scheduler, scaler=self.scaler, epoch=epoch,
                                                 global_step=global_step, best_metrics=best,
                                                 identity=self.identity(), history=self.history,
                                                 record_set_hashes=self.record_set_hashes(),
                                                 git_commit=git_commit(Path.cwd()), resume_lineage=self.resume_lineage)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            save_checkpoint(last_path, model=self.model, optimizer=self.optimizer, scheduler=scheduler,
                            scaler=self.scaler, epoch=epoch, global_step=global_step, best_metrics=best,
                            identity=self.identity(), history=self.history,
                            record_set_hashes=self.record_set_hashes(), git_commit=git_commit(Path.cwd()),
                            resume_lineage=self.resume_lineage)
            atomic_json_write(self.run_root / "logs" / "history.json", self.history)
            if progress: progress({"stage": "epoch", "epoch": epoch, **{k: round(v, 5) for k, v in validation.items()}})
            if (bool(stopping["enabled"]) and epoch + 1 >= int(stopping["min_epochs"])
                    and epochs_without_improvement >= int(stopping["patience_epochs"])):
                stop_reason = f"early_stopped_patience_{stopping['patience_epochs']}"
                break
        return {"run_id": run_id, "epochs_run": len(self.history), "epochs_configured": epochs,
                "global_step": global_step, "stop_reason": stop_reason, "best": best,
                "history": self.history, "device": self.device, "amp": self.use_amp,
                "identity": self.identity(), "record_set_hashes": self.record_set_hashes(),
                "loss_manifest": loss_manifest(self.config.get("loss")),
                "checkpoints": {"best": best_path.name, "last": last_path.name,
                                "best_sha256": best.get("sha256"),
                                "last_sha256": sha256_file(last_path) if last_path.is_file() else None},
                "train_pairs": len(self.train_pairs), "validation_pairs": len(self.validation_pairs),
                "ll_invariant_tolerance": LL_INVARIANT_TOLERANCE,
                "source_isolation": self.audit.report()}
