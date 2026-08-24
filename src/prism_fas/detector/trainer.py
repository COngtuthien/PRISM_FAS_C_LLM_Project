"""The M9 stage trainer: G1, G2, G5, G6.

Spec section 11 (Table 39, Table 40). M9 runs exactly four stages:

    G1  Baseline warm-up      source real live/spoof      -> binary detector checkpoint
    G2  Real manifold init    source_train live embeddings-> prototypes + covariance
    G5  Mixed training        real + accepted synthetic   -> full PRISM checkpoint
    G6  Source calibration    source_dev predictions      -> thresholds.json

G7 (target prediction) and G8 (final scoring) are M10 and are unreachable from
here: `target_test` is never opened, and `check_stage_transition` refuses any stage
outside the frozen flow.

Warm-up schedule (Table 37 "3 detector + 2 manifold" read together with section 9.3
"K-means after detector warm-up"): 3 G1 epochs on real data with the prototypes
absent, then the G2 K-means initialization, then 2 manifold warm-up epochs in which
the manifold terms are live, then the 30 G5 mixed epochs.

`source_dev` is read only inside `validate()` and `run_g6()`, under `no_grad`, and
never produces an optimization gradient. This module never imports modal.
"""
from __future__ import annotations
import hashlib, json, math, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
from prism_fas.utils.core import atomic_json_write
from .checkpoint import (STAGE_ORDER, M9CheckpointError, RunIdentity, StageLineage,
                         apply_checkpoint, check_stage_transition, check_status_transition,
                         checkpoint_summary, config_hash, git_commit, load_checkpoint,
                         save_checkpoint)
from .contracts import LIVE, REGION_COUNT, DetectorBatch
from .dataset import (DOMAINS as SOURCE_DOMAINS, M9TrainingDataset, M9ValidationDataset,
                      batch_composition, domain_composition)
from .heads import read_recipe_text_cache
from .losses import (DEFAULT_CLEAN_CAP, DEFAULT_MIL_TEMPERATURE, DEFAULT_PROMPT_TEMPERATURE,
                     DEFAULT_WEIGHTS, LOSS_NAMES, compute_losses, loss_contract_identity,
                     loss_graph_delta)
from .manifold import initialize_prototypes, write_prototypes_npz
from .prism_detector import DetectorConfig, PRISMDetector, build_detector
from .sampler import DEFAULT_COMPOSITION, BatchContract, M9BatchSampler
from .variant import ResolvedExperimentVariant

TRAINER_SCHEMA_VERSION = "m9-trainer-v1"
# Loss terms that need initialized prototypes. Absent during G1 by construction,
# not by a magic zero.
MANIFOLD_TERMS = ("L_real", "L_out", "L_clean")
# G1/G2 read real data only (Table 39), so the synthetic terms are inactive there.
SYNTHETIC_TERMS = ("L_cls_syn", "L_local", "L_out", "L_clean", "L_prompt")


class TrainerError(RuntimeError):
    """The M9 trainer cannot proceed as declared."""


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed python, numpy and torch from the declared run seed.

    Reuses the M5 seeding helper rather than adding a second convention.
    """
    from prism_fas.train.seed import seed_everything as _seed
    _seed(int(seed))
    return {"seed": int(seed)}


def configure_determinism() -> dict[str, Any]:
    """cuDNN deterministic, benchmark off (`configs/train/m9_reference.yaml`)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"cudnn_deterministic": True, "cudnn_benchmark": False}


@dataclass(frozen=True)
class M9TrainingConfig:
    """Spec section 10.3 (Table 37) plus the declared M9 schedule."""
    run_id: str = "m9_reference_seed20260806"
    seed: int = 20260806
    # Table 37: warm-up 3 detector + 2 manifold, then 30 detector epochs.
    warmup_detector_epochs: int = 3
    manifold_warmup_epochs: int = 2
    mixed_epochs: int = 30
    steps_per_epoch: int = 45
    accumulation_steps: int = 1
    backbone_lr: float = 1.0e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_fraction: float = 0.05
    min_lr_scale: float = 0.0
    gradient_clip_norm: float = 1.0
    amp: bool = True
    ema_enabled: bool = False              # Table 37 makes EMA optional; M9 reports it OFF
    checkpoint_every_steps: int = 250
    validate_every_epochs: int = 1
    validation_batch_size: int = 32
    validation_limit: int | None = None
    selection_metric: str = "source_dev/acer"
    tie_break_metric: str = "source_dev/bpcer"
    calibration_metric: str = "source_dev/nll"
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    clean_cap: float = DEFAULT_CLEAN_CAP
    mil_temperature: float = DEFAULT_MIL_TEMPERATURE
    prompt_temperature: float = DEFAULT_PROMPT_TEMPERATURE
    prototype_seed: int = 20260806
    prototype_batch_size: int = 28
    # The M10 switch set. Defaults to the frozen B08 reference, so an M9 call site
    # that never mentions a variant gets exactly the M9 reference training config.
    variant: ResolvedExperimentVariant = field(default_factory=ResolvedExperimentVariant.reference)
    # The content identity of the synthetic bank this row actually opens. Empty means
    # the frozen M8 v3 bank, which is what `frozen_inputs` already pins for every
    # structured row — so the reference config hash is unchanged. A row that opens a
    # DIFFERENT bank (A02's random-operator artifact) carries its identity here, and
    # the config hash therefore differs. The identity is also bound independently as
    # `RunIdentity.m8_bank_identity` and inside `dataset_contract_identity`; this
    # field is what puts it in the TRAINING CONFIG identity as well, so an A02
    # checkpoint can never be resumed against the structured bank.
    synthetic_bank_identity: str = ""
    # The §19 protocol's source domains. Both is the Version-B reference and is
    # carried into the config payload only when a protocol restricts it, so a P1
    # or P2 row hashes differently while every inherited row keeps the config
    # hash its checkpoints were written with.
    source_domains: tuple[str, ...] = SOURCE_DOMAINS

    @property
    def total_epochs(self) -> int:
        """Identical for EVERY variant: 3 + 2 + 30.

        A variant without a manifold has no prototypes to warm up, but it still
        trains for the same number of epochs — its two manifold warm-up epochs are
        ordinary G1 epochs. Shortening the schedule for the simpler baselines would
        confound every Table 59 comparison with a difference in training length.
        """
        return self.warmup_detector_epochs + self.manifold_warmup_epochs + self.mixed_epochs

    # `run_id` is a run LABEL, not a configuration value. Spec section 11.1 makes the
    # run id the thing you change when the science changes, so folding it into the
    # config hash would make every identity derived from that hash — including the
    # prototype identity — differ between two runs of the identical configuration.
    # `variant` is excluded from the raw dump and re-entered below as a DELTA, so the
    # M9 reference configuration keeps the config hash its checkpoints were written
    # with while every ablation gets a different one.
    HASH_EXCLUDED_FIELDS = ("run_id", "variant", "synthetic_bank_identity", "source_domains")

    def payload(self, *, include_run_id: bool = False) -> dict[str, Any]:
        body = {key: value for key, value in vars(self).items()
                if key not in ("variant", "synthetic_bank_identity", "source_domains")
                and (include_run_id or key not in self.HASH_EXCLUDED_FIELDS)}
        body["schema_version"] = TRAINER_SCHEMA_VERSION
        body["total_epochs"] = self.total_epochs
        delta = training_delta(self.variant)
        if delta: body["variant_training"] = delta
        # Carried only when this row opens a bank other than the frozen M8 v3 one, so
        # every structured row keeps the config hash its checkpoints were written with.
        if self.synthetic_bank_identity:
            body["synthetic_bank_identity"] = str(self.synthetic_bank_identity)
        if tuple(self.source_domains) != SOURCE_DOMAINS:
            body["source_domains"] = list(self.source_domains)
        return body

    def hash(self) -> str: return config_hash(self.payload())

    def resolved(self) -> dict[str, Any]:
        """The full resolved config, run label included, for `run.json`."""
        return {**self.payload(include_run_id=True), "config_hash": self.hash()}


def training_delta(variant: ResolvedExperimentVariant) -> dict[str, Any]:
    """The training switches on which `variant` differs from the B08 reference.

    These change WHAT IS OPTIMIZED without changing a single parameter shape, so an
    architecture hash cannot catch them. Binding them into the training config hash
    is what makes a B08 checkpoint refuse to resume as A04.
    """
    reference = ResolvedExperimentVariant.reference().training_payload()
    return {key: value for key, value in variant.training_payload().items()
            if reference.get(key) != value}


def stage_for_epoch(epoch: int, config: M9TrainingConfig) -> str:
    """The frozen schedule, expressed once, for the variant's declared flow.

    With a manifold: 3 G1 epochs, 2 G2 manifold warm-up epochs, then the mixed
    epochs. Without one there is nothing to warm up, so those two epochs are
    ordinary G1 epochs and the total schedule length is unchanged.
    """
    warmup = config.warmup_detector_epochs
    if not config.variant.has_manifold:
        return "G1" if epoch < warmup + config.manifold_warmup_epochs else "G5"
    if epoch < warmup: return "G1"
    if epoch < warmup + config.manifold_warmup_epochs: return "G2"
    return "G5"


def enabled_terms(stage: str, variant: ResolvedExperimentVariant | None = None) -> dict[str, bool]:
    """Which declared losses are live in a stage, for this variant.

    G1: prototypes are absent and no synthetic sample is in the batch, so the
    manifold and synthetic terms are switched OFF rather than silently evaluating
    to a meaningless constant.
    G2: prototypes exist; the manifold terms come on. Synthetic still absent.
    G5: everything the variant declares — which for `synthetic: none` still excludes
    every synthetic term, because those samples do not exist for that row.
    """
    return (variant or ResolvedExperimentVariant.reference()).stage_loss_terms(stage)


def batch_contract_for(stage: str, config: M9TrainingConfig) -> BatchContract:
    """Table 36 for a mixed G5; the real-only composition otherwise.

    G1/G2 read "Source real live/spoof" (Table 39). Removing the synthetic quarter
    and keeping Table 36's 1:1 real live/spoof ratio at batch size 32 gives 16/16.
    That split is SPEC_UNDERSPECIFIED and declared in DECISIONS.md.

    A variant with `synthetic: none` uses the SAME real-only composition for G5. It
    is not given a fake 12/12/8 batch with eight fabricated synthetic slots, and its
    synthetic loss terms are structurally inactive rather than evaluating to a
    constant placeholder. The real live/spoof partitions stay balanced according to
    the row's own sampler policy.
    """
    variant = config.variant
    domains = tuple(config.source_domains)
    if stage in ("G1", "G2") or not variant.uses_synthetic:
        return BatchContract(real_live=16, real_spoof=16, synthetic=0, phase="real_only",
                             accumulation_steps=config.accumulation_steps,
                             domain_balance=variant.domain_balance, domains=domains,
                             require_both_routes=False, routes=variant.synthetic_routes or ("physics", "gpat"))
    return BatchContract(real_live=DEFAULT_COMPOSITION["real_live"],
                         real_spoof=DEFAULT_COMPOSITION["real_spoof"],
                         synthetic=DEFAULT_COMPOSITION["synthetic"],
                         accumulation_steps=config.accumulation_steps,
                         domain_balance=variant.domain_balance, domains=domains,
                         require_both_routes=variant.requires_both_routes,
                         routes=variant.synthetic_routes)


@dataclass
class M9Trainer:
    """Owns the model, data, optimizer and the G1/G2/G5/G6 state machine."""
    config: M9TrainingConfig
    detector_config: DetectorConfig
    package_root: Path
    bank_root: Path
    recipe_bank_root: Path
    run_root: Path
    cache_root: Path
    weight_root: Path
    loader_config_path: Path
    device: str = "cpu"
    text_cache_path: Path | None = None
    validation_limit: int | None = None
    # Opt-in, and only used to MINT the frozen recipe text cache artifact. A
    # reference run always loads the uploaded artifact instead.
    allow_text_cache_build: bool = False
    progress: Callable[[dict[str, Any]], None] | None = None
    # The pinned identity of the synthetic bank this row opens. `None` keeps the
    # frozen M8 v3 pin; the A02 random-operator row passes its own bank's pin, and
    # neither is ever a fallback for the other.
    bank_identity: str | None = None
    bank_id: str | None = None
    # An ALREADY-VERIFIED synthetic bank reader. Version C's C7/C8 pass one arm's
    # frozen C6 matched bank here (`detector.c6_bank.C6MatchedBankReader`), which
    # is a different artifact from the M8 v3 export and is opened by its own
    # fail-closed reader. `None` keeps the inherited behaviour exactly: the M8 v3
    # bank is opened from `bank_root` against its pinned identity.
    synthetic_bank: Any = None

    def __post_init__(self) -> None:
        from prism_fas.data.loader.config import load_loader_config
        from prism_fas.detector.pretrained import SigLIP2Artifacts, resolve_convnext_weight, sha256_file
        from prism_fas.recipes.bank import load_bank
        for name in ("package_root", "bank_root", "recipe_bank_root", "run_root", "cache_root",
                     "weight_root", "loader_config_path"):
            setattr(self, name, Path(getattr(self, name)))
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.loader_config = load_loader_config(self.loader_config_path)
        # Seed BEFORE the model is built (spec section 12.7). Without this the head
        # and fusion weights come from ambient RNG state, so two independent runs of
        # the same config produce different initial embeddings and therefore
        # different prototypes — which is exactly what the twice-run prototype
        # initialization check is there to catch.
        self.seed_state = seed_everything(int(self.config.seed))
        self.determinism = configure_determinism()
        self.siglip = SigLIP2Artifacts.resolve(self.weight_root)
        self.convnext_path = resolve_convnext_weight(self.weight_root)
        self.convnext_sha256 = sha256_file(self.convnext_path)
        bank = load_bank(self.recipe_bank_root)
        self.recipe_bank_identity = str(bank["lock"]["bank_content_identity_sha256"])
        self.recipe_ids = tuple(recipe.recipe_id for recipe in bank["recipes"])
        self.text_cache = self._text_cache()
        self.variant = self.config.variant.validate().require_executable()
        self.dataset = M9TrainingDataset(self.package_root, self.bank_root, self.loader_config,
                                         cache_root=self.cache_root, recipe_ids=self.recipe_ids,
                                         variant=self.variant,
                                         bank_identity=self.bank_identity, bank_id=self.bank_id,
                                         bank=self.synthetic_bank,
                                         domains=self.config.source_domains,
                                         progress=lambda done, total: self._emit(
                                             {"stage": "prepare", "cache_progress": done, "total": total}))
        self.text_embeddings = self.text_cache.tensor(device=self.device)
        self.model: PRISMDetector = build_detector(
            self.detector_config, text_embeddings=self.text_cache.tensor(), siglip=self.siglip,
            local_weight_file=self.convnext_path, text_cache_identity=self.text_cache.identity,
            device=self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(backbone_lr=self.config.backbone_lr, head_lr=self.config.head_lr,
                                        weight_decay=self.config.weight_decay),
            betas=tuple(self.config.betas))
        self.total_steps = self.config.total_epochs * self.config.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.amp_dtype, self.amp_enabled = self._amp()
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.identity = self._identity()
        self.stages = self.variant.required_stages()
        self.lineage = StageLineage(order=self.stages)
        self.epoch, self.global_step, self.step_in_epoch, self.stage = 0, 0, 0, "G1"
        self.status = "PENDING"
        self.best_metrics: dict[str, Any] = {}
        self.prototype_identity = ""
        self.history: list[dict[str, Any]] = []
        self._validation: M9ValidationDataset | None = None
        self.samplers = {stage: self._sampler(stage) for stage in ("G1", "G5")}
        self.samplers["G2"] = self.samplers["G1"]
        # AdamW receives only non-empty groups (B01 has no trainable backbone), so
        # a learning rate is read by NAME rather than by a positional index that
        # would silently point at the wrong group.
        self.lr_group = {str(group.get("name", index)): index
                         for index, group in enumerate(self.optimizer.param_groups)}

    # --- setup helpers ------------------------------------------------------
    def _text_cache(self) -> Any:
        """Load the FROZEN recipe text cache artifact.

        It is never rebuilt during training. Encoding the 128 descriptions is
        deterministic within one environment but not bit-identical across
        torch/transformers builds or across CPU and GPU, so a silent rebuild would
        hand the run a different content identity for the same science — which is
        exactly the drift the identity guard exists to catch. `allow_text_cache_build`
        is opt-in and used only to MINT the artifact, never inside a reference run.
        """
        from .heads import build_recipe_text_cache, resolve_recipe_text_cache, write_recipe_text_cache
        if self.text_cache_path is not None:
            return read_recipe_text_cache(Path(self.text_cache_path))
        try:
            return resolve_recipe_text_cache(self.weight_root)
        except Exception:                             # noqa: BLE001 - reported below if not allowed
            if not self.allow_text_cache_build: raise
        cache = build_recipe_text_cache(self.recipe_bank_root, self.siglip, device=self.device)
        write_recipe_text_cache(self.cache_root / "recipe_text_cache.npz", cache)
        return cache

    def _amp(self) -> tuple[Any, bool]:
        """Table 37: bf16 when supported, else fp16. CPU runs fp32."""
        if not self.config.amp or self.device != "cuda": return torch.float32, False
        return (torch.bfloat16, True) if torch.cuda.is_bf16_supported() else (torch.float16, True)

    def _lr_lambda(self, step: int) -> float:
        """5 % linear warm-up then cosine decay (Table 37)."""
        warmup = max(1, int(self.total_steps * float(self.config.warmup_fraction)))
        if step < warmup: return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return float(self.config.min_lr_scale + (1.0 - self.config.min_lr_scale) * cosine)

    def _sampler(self, stage: str) -> M9BatchSampler:
        return M9BatchSampler(real_live=self.dataset.real_live, real_spoof=self.dataset.real_spoof,
                              synthetic_routes=self.dataset.synthetic_routes,
                              contract=batch_contract_for(stage, self.config), seed=self.config.seed,
                              steps_per_epoch=self.config.steps_per_epoch,
                              identity=self.dataset.package_identity)

    def _identity(self) -> RunIdentity:
        return RunIdentity(
            source_package_identity=self.dataset.package_identity,
            m8_bank_identity=self.dataset.bank.identity,
            architecture_identity=self.model.architecture_identity(),
            siglip2_identity=self.siglip.identity(),
            recipe_text_cache_identity=self.text_cache.identity,
            config_hash=self.config.hash(),
            loss_contract_hash=loss_contract_identity(
                self.config.loss_weights, {"clean_cap": self.config.clean_cap,
                                           "mil_temperature": self.config.mil_temperature,
                                           "prompt_temperature": self.config.prompt_temperature,
                                           **loss_graph_delta(self.variant)}),
            batch_contract_hash=batch_contract_for("G5", self.config).identity(),
            dataset_contract_identity=self.dataset.contract_identity(),
            convnext_weight_sha256=self.convnext_sha256,
            m7_recipe_bank_identity=self.recipe_bank_identity,
            region_prior_cache_identity=self.dataset.prior_cache.identity,
            attack_mask_cache_identity=self.dataset.attack_cache.identity).validate()

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.progress: self.progress(payload)

    # --- run artifacts ------------------------------------------------------
    def stage_root(self, stage: str) -> Path:
        root = self.run_root / "stages" / stage
        root.mkdir(parents=True, exist_ok=True)
        return root

    def write_stage_state(self, stage: str, status: str, extra: dict[str, Any] | None = None) -> Path:
        path = self.stage_root(stage) / "stage_state.json"
        atomic_json_write(path, {"schema_version": TRAINER_SCHEMA_VERSION, "stage": stage,
                                 "status": status, "run_id": self.config.run_id,
                                 "epoch": self.epoch, "global_step": self.global_step,
                                 "identity": self.identity.payload(), **(extra or {})})
        return path

    def write_input_hashes(self, stage: str, payload: dict[str, Any]) -> Path:
        path = self.stage_root(stage) / "input_hashes.json"
        atomic_json_write(path, {"stage": stage, **payload})
        return path

    def write_output_hashes(self, stage: str, payload: dict[str, Any]) -> Path:
        path = self.stage_root(stage) / "output_hashes.json"
        atomic_json_write(path, {"stage": stage, **payload})
        return path

    def write_resolved_config(self, stage: str) -> Path:
        """Table 40 requires a `resolved_config.yaml` beside every stage's state."""
        import yaml
        path = self.stage_root(stage) / "resolved_config.yaml"
        payload = {"stage": stage, "run_id": self.config.run_id,
                   "training": self.config.resolved(), "detector": self.detector_config.payload(),
                   "batch_contract": batch_contract_for(stage, self.config).payload(),
                   "variant": self.variant.payload(),
                   "active_loss_terms": enabled_terms(stage, self.variant),
                   "identity": self.identity.payload()}
        path.write_text(yaml.safe_dump(payload, sort_keys=True, default_flow_style=False),
                        encoding="utf-8")
        return path

    def write_run_json(self) -> Path:
        """The run-level record. Spec section 10.3 requires `ema_enabled` to be
        reported explicitly, and every stage in the lineage gets its resolved
        config backfilled so a resumed run leaves the same artifacts as a fresh one.
        """
        for entry in self.lineage.payload(): self.write_resolved_config(str(entry["stage"]))
        path = self.run_root / "run.json"
        atomic_json_write(path, self.run_summary())
        return path

    def append_metrics(self, stage: str, record: dict[str, Any]) -> None:
        path = self.stage_root(stage) / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n")

    def stage_inputs(self, stage: str) -> dict[str, Any]:
        """What this stage is allowed to consume, hashed. A stage never reads a
        future stage's artifact, and G5 never reads the target package."""
        payload = {"source_package_identity": self.dataset.package_identity,
                   "architecture_identity": self.model.architecture_identity(),
                   "siglip2_identity": self.siglip.identity(),
                   "recipe_text_cache_identity": self.text_cache.identity,
                   "batch_contract": batch_contract_for(stage, self.config).payload(),
                   "config_hash": self.config.hash(), "target_test_opened": False}
        payload["variant_identity"] = self.variant.identity()
        if stage in ("G2", "G5"):
            payload["live_population_identity"] = self.dataset.population_identity(self.dataset.live_positions())
        if stage == "G5" and self.variant.uses_synthetic:
            payload["m8_bank_identity"] = self.dataset.bank.identity
            payload["m8_bank_id"] = self.dataset.bank.bank_id
            payload["synthetic_routes"] = list(self.variant.synthetic_routes)
        if stage == "G5":
            payload["prototype_identity"] = self.prototype_identity
        if stage == "G6":
            payload["prototype_identity"] = self.prototype_identity
            payload["source_dev_opened"] = True
        return payload

    # --- training -----------------------------------------------------------
    def losses_for(self, output: Any, batch: DetectorBatch, stage: str) -> Any:
        # The adapted text matrix when the variant declares `prompt: adapter`, the
        # frozen cache otherwise. `L_prompt` must score against the same matrix the
        # head scored against, or the InfoNCE target and the fusion evidence would
        # disagree.
        text = self.model.text_matrix()
        return compute_losses(output, batch, self.model.manifold, weights=self.config.loss_weights,
                              text_embeddings=text if text is not None else self.text_embeddings,
                              clean_cap=self.config.clean_cap,
                              mil_temperature=self.config.mil_temperature,
                              prompt_temperature=self.config.prompt_temperature,
                              enabled=enabled_terms(stage, self.variant), variant=self.variant)

    def train_step(self, plan: Any, stage: str) -> dict[str, Any]:
        """One optimizer step over the plan, split into micro-batches that each
        carry the exact declared composition."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro = plan.microbatches(self.config.accumulation_steps)
        accumulated: dict[str, float] = {}
        applicable: dict[str, int] = {}
        total_value = 0.0
        composition = {"real_live": 0, "real_spoof": 0, "synthetic_spoof": 0}
        for part in micro:
            batch = self.dataset.batch_from_plan(part).to(self.device)
            for key, value in batch_composition(batch).items(): composition[key] += value
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled):
                output = self.model(batch)
                result = self.losses_for(output, batch, stage)
            loss = result.total / len(micro)
            self.scaler.scale(loss).backward() if self.scaler.is_enabled() else loss.backward()
            for name, value in result.terms.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value.detach()) / len(micro)
            for name, value in result.applicable.items():
                applicable[name] = applicable.get(name, 0) + int(value)
            total_value += float(result.total.detach()) / len(micro)
            # Prototype update: LIVE, non-synthetic, valid sites only, under
            # no_grad inside `RealManifold.update`. A variant with no manifold has
            # no prototype state to move.
            if stage in ("G2", "G5") and self.model.manifold is not None:
                sites = (output.aux or {}).get("manifold_embeddings", output.region_embeddings)
                self.model.manifold.update(sites.detach(),
                                           is_live=(batch.label == LIVE) & ~batch.is_synthetic,
                                           is_synthetic=batch.is_synthetic,
                                           region_valid=output.region_valid)
        if self.scaler.is_enabled(): self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.grad is not None],
            float(self.config.gradient_clip_norm)))
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer); self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        if not math.isfinite(total_value): raise TrainerError(f"L_total is not finite at step {self.global_step}")
        learning_rates = {f"lr_{name}": self.optimizer.param_groups[index]["lr"]
                          for name, index in self.lr_group.items()}
        return {"stage": stage, "epoch": self.epoch, "global_step": self.global_step,
                "step_in_epoch": plan.step, "L_total": total_value, "grad_norm": grad_norm,
                **learning_rates,
                "composition": composition, "microbatches": len(micro),
                **{name: value for name, value in sorted(accumulated.items())},
                **{f"applicable/{name}": value for name, value in sorted(applicable.items())}}

    def run_epoch(self, stage: str, *, start_step: int = 0, max_steps: int | None = None) -> list[dict[str, Any]]:
        """Run `epoch` from `start_step`. `step_in_epoch` is the resume position:
        a resumed run continues the same epoch's plan sequence rather than
        restarting it."""
        sampler = self.samplers[stage]
        records: list[dict[str, Any]] = []
        self.step_in_epoch = int(start_step)
        for plan in sampler.iter_epoch(self.epoch, start_step=start_step):
            record = self.train_step(plan, stage)
            self.step_in_epoch += 1
            self.append_metrics(stage, record)
            records.append(record)
            self._emit({"stage": stage, "epoch": self.epoch, "step": self.global_step,
                        "L_total": record["L_total"]})
            if self.config.checkpoint_every_steps and self.global_step % self.config.checkpoint_every_steps == 0:
                self.save("last")
            if max_steps is not None and len(records) >= max_steps: break
        return records

    # --- stages -------------------------------------------------------------
    def enter_stage(self, stage: str) -> None:
        # Against THIS RUN's declared flow, not the full G1->G2->G5->G6 default: a
        # variant with `manifold: off` legitimately goes G1 -> G5, and checking it
        # against the default here refused the transition before the lineage — which
        # carries the right flow — ever saw it.
        check_stage_transition(self.lineage.current, stage, order=self.stages)
        self.lineage.enter(stage, input_hashes=self.stage_inputs(stage))
        self.stage = stage
        if self.status == "PENDING": self.status = check_status_transition(self.status, "RUNNING")
        self.write_stage_state(stage, "RUNNING")
        self.write_input_hashes(stage, self.stage_inputs(stage))
        self.write_resolved_config(stage)

    def complete_stage(self, stage: str, output_hashes: dict[str, Any]) -> None:
        self.lineage.complete(stage, output_hashes=output_hashes)
        self.write_output_hashes(stage, output_hashes)
        self.write_stage_state(stage, "COMPLETED", {"output_hashes": output_hashes})
        self.write_run_json()

    def g1_epochs(self) -> int:
        """3 detector warm-up epochs, plus the 2 manifold warm-up epochs when this
        variant has no manifold to warm up. The total schedule length is the same for
        every variant, so no Table 59 comparison is confounded by training length."""
        return self.config.warmup_detector_epochs + (
            0 if self.variant.has_manifold else self.config.manifold_warmup_epochs)

    def run_g1(self) -> dict[str, Any]:
        """Baseline warm-up on real source data. Prototypes are absent."""
        self.enter_stage("G1")
        started = time.time()
        planned = self.g1_epochs()
        for _ in range(planned):
            self.run_epoch("G1")
            self.epoch += 1
            self.step_in_epoch = 0
        sha = self.save("last")
        output = {"epochs": planned, "global_step": self.global_step,
                  "checkpoint_sha256": sha, "prototypes_initialized": False,
                  "manifold_declared": self.variant.has_manifold,
                  "seconds": round(time.time() - started, 2)}
        self.complete_stage("G1", output)
        return output

    @torch.no_grad()
    def collect_live_embeddings(self, *, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Regional embeddings for every `source_train` LIVE sample.

        Only `label == live` rows of `source_train` are read: no real spoof, no
        synthetic, no `source_dev`, no target (spec section 9.3).
        """
        self.model.eval()
        positions = self.dataset.live_positions()
        size = int(batch_size or self.config.prototype_batch_size)
        embeddings: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        datasets: dict[str, int] = {}
        for start in range(0, len(positions), size):
            chunk = positions[start:start + size]
            batch = self.dataset.live_batch(chunk).to(self.device)
            if int(batch.label.max()) != LIVE:
                raise TrainerError("a non-live sample reached prototype initialization")
            if bool(batch.is_synthetic.any()):
                raise TrainerError("a synthetic sample reached prototype initialization")
            output = self.model(batch)
            # The MANIFOLD sites, which are the nine region embeddings for a regional
            # manifold and one pooled image embedding for a global center.
            sites = (output.aux or {}).get("manifold_embeddings", output.region_embeddings)
            embeddings.append(sites.detach().float().cpu().numpy())
            valid.append(output.region_valid.detach().cpu().numpy())
            for index in batch.dataset_id.tolist():
                name = batch.datasets[index]
                datasets[name] = datasets.get(name, 0) + 1
            self._emit({"stage": "G2", "collected": start + len(chunk), "total": len(positions)})
        stacked = np.concatenate(embeddings, axis=0)
        mask = np.concatenate(valid, axis=0)
        audit = {"live_samples": len(positions), "datasets": dict(sorted(datasets.items())),
                 "valid_per_region": {name: int(mask[:, index].sum())
                                      for index, name in enumerate(self.model.manifold.region_names)},
                 "population_identity_sha256": self.dataset.population_identity(positions),
                 "spoof_used": False, "synthetic_used": False, "source_dev_used": False,
                 "target_used": False}
        return stacked, mask, audit

    def run_g2(self, *, prototypes_path: Path | None = None) -> dict[str, Any]:
        """K-means prototype initialization plus the manifold warm-up epochs.

        Only variants that declare a manifold have this stage at all; the stage
        machine refuses it for the others rather than running an empty one.
        """
        if not self.variant.has_manifold:
            raise TrainerError("this variant declares manifold=off; G2 is not part of its declared flow")
        self.enter_stage("G2")
        started = time.time()
        embeddings, valid, audit = self.collect_live_embeddings()
        state = initialize_prototypes(embeddings, valid, k=self.variant.prototype_k,
                                      epsilon=self.detector_config.covariance_epsilon,
                                      seed=self.config.prototype_seed,
                                      region_names=self.model.manifold.region_names)
        self.model.manifold.load_state(state)
        population = self.dataset.population_identity(self.dataset.live_positions())
        self.prototype_identity = state.identity(config_hash=self.config.hash(),
                                                 population_identity=population)
        # `normalize(d_r)` scale: the median live distance per region on the same
        # population, set once here and frozen (contract section 7).
        self.model.set_distance_scale(self._distance_scale(embeddings, valid), freeze=True)
        target = Path(prototypes_path or (self.stage_root("G2") / "prototypes.npz"))
        export = write_prototypes_npz(target, state, config_hash=self.config.hash(),
                                      population_identity=population,
                                      feature_identity=self.model.architecture_identity())
        for _ in range(self.config.manifold_warmup_epochs):
            self.run_epoch("G2")
            self.epoch += 1
            self.step_in_epoch = 0
        sha = self.save("last")
        output = {"prototype_identity_sha256": self.prototype_identity, "prototypes": export,
                  "population": audit, "k": self.variant.prototype_k,
                  "manifold_scope": self.variant.manifold_scope,
                  "manifold_sites": list(self.model.manifold.region_names),
                  "distance_scale": self.model.distance_scale.detach().cpu().tolist(),
                  "manifold_warmup_epochs": self.config.manifold_warmup_epochs,
                  "checkpoint_sha256": sha, "global_step": self.global_step,
                  "seconds": round(time.time() - started, 2)}
        self.complete_stage("G2", output)
        return output

    def _distance_scale(self, embeddings: np.ndarray, valid: np.ndarray) -> torch.Tensor:
        """The per-region median live distance that `normalize(d_r)` divides by,
        computed on the same live population that built the prototypes and frozen
        from here on. Uses the module's declared distance scale convention, so the
        normalizer and the losses measure the same quantity."""
        convention = self.model.manifold.distance_scale_convention
        sites = int(embeddings.shape[1])
        scale = np.ones(sites, dtype=np.float32)
        for region in range(sites):
            rows = embeddings[valid[:, region], region, :]
            if rows.shape[0] < 2: continue
            centre = rows.mean(axis=0)
            variance = np.maximum(rows.var(axis=0), self.detector_config.covariance_epsilon)
            distances = (((rows - centre) ** 2) / variance).sum(axis=1)
            if convention == "mean_per_dimension": distances = distances / float(rows.shape[1])
            scale[region] = max(float(np.median(distances)), 1e-6)
        return torch.from_numpy(scale)

    def run_g5(self, *, epochs: int | None = None) -> dict[str, Any]:
        """Mixed training: real + the accepted synthetic bank."""
        self.enter_stage("G5")
        if self.variant.has_manifold and not bool(self.model.manifold.initialized):
            raise TrainerError("G5 requires initialized prototypes; run G2 first")
        started = time.time()
        planned = int(epochs if epochs is not None else self.config.mixed_epochs)
        for index in range(planned):
            self.run_epoch("G5")
            self.epoch += 1
            self.step_in_epoch = 0
            if self.config.validate_every_epochs and (index + 1) % self.config.validate_every_epochs == 0:
                metrics = self.validate()
                self.append_metrics("G5", {"kind": "validation", "epoch": self.epoch, **metrics})
                if self.is_better(metrics):
                    self.best_metrics = {**metrics, "epoch": self.epoch, "global_step": self.global_step}
                    self.save("best")
        sha = self.save("last")
        output = {"epochs": planned, "global_step": self.global_step, "checkpoint_sha256": sha,
                  "best_metrics": dict(self.best_metrics), "seconds": round(time.time() - started, 2)}
        self.complete_stage("G5", output)
        return output

    def run_g6(self, *, checkpoint: Path | None = None) -> dict[str, Any]:
        """Source calibration on `source_dev` only. No gradient, no target."""
        self.enter_stage("G6")
        started = time.time()
        from prism_fas.detector.checkpoint import sha256_file
        from prism_fas.train.calibration import calibrate_source_dev
        logits, targets = self.source_dev_logits()
        # The prediction hash binds the exact source_dev predictions the thresholds
        # were fitted on, so a threshold file can never be paired with a different
        # prediction set.
        prediction_hash = hashlib.sha256(
            json.dumps([[format(float(value), ".12g") for value in logits.tolist()],
                        [int(value) for value in targets.tolist()]],
                       separators=(",", ":")).encode("utf-8")).hexdigest()
        path = Path(checkpoint) if checkpoint else None
        calibration = calibrate_source_dev(
            logits, targets,
            checkpoint_sha=sha256_file(path) if path and path.is_file() else "",
            package_identity=self.dataset.package_identity,
            prediction_hash=prediction_hash, run_root=self.run_root)
        threshold_path = self.stage_root("G6") / "thresholds.json"
        atomic_json_write(threshold_path, calibration)
        output = {"thresholds": calibration, "samples": int(targets.size),
                  "source_dev_prediction_hash": prediction_hash,
                  "source_dev_opened": True, "target_test_opened": False,
                  "produced_gradient": False, "seconds": round(time.time() - started, 2)}
        self.complete_stage("G6", output)
        return output

    # --- validation ---------------------------------------------------------
    def validation(self) -> M9ValidationDataset:
        if self._validation is None:
            self._validation = M9ValidationDataset(
                self.package_root, self.loader_config, cache_root=self.cache_root,
                limit=self.validation_limit if self.validation_limit is not None else self.config.validation_limit,
                domains=self.config.source_domains,
                progress=lambda done, total: self._emit({"stage": "validate", "cache_progress": done, "total": total}))
        return self._validation

    @torch.no_grad()
    def source_dev_logits(self, positions: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """`source_dev` forward under `no_grad`. It cannot create a gradient, and
        the caller never backpropagates it.

        `output.global_logit` is the DECISION logit slot, whatever the variant
        names it: for a `glr_concat` Track-R row the same tensor is
        `fused_logit_R`, produced by the fusion classifier over the concatenated
        global/local/region summary. Reading a different field for Track R would
        calibrate one quantity and threshold another — the §16.2 defect
        `calibration_identity_guard` exists to refuse — so the slot is read once
        here and its NAME travels with the metrics.
        """
        self.model.eval()
        logits: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        validation = self.validation()
        stream = (validation.batches(self.config.validation_batch_size) if positions is None
                  else (validation.batch(list(positions)[start:start + self.config.validation_batch_size])
                        for start in range(0, len(list(positions)), self.config.validation_batch_size)))
        for batch in stream:
            moved = batch.to(self.device)
            output = self.model(moved)
            logits.append(output.global_logit.detach().float().cpu().numpy().reshape(-1))
            targets.append(moved.label.detach().cpu().numpy().reshape(-1))
        if not logits:
            raise TrainerError("source_dev produced no rows for the requested positions")
        return np.concatenate(logits), np.concatenate(targets)

    @property
    def decision_logit_name(self) -> str:
        return self.variant.decision_logit_name

    @property
    def decision_score_name(self) -> str:
        return self.variant.decision_score_name

    @staticmethod
    def _dev_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        from prism_fas.train.metrics import negative_log_likelihood, roc_auc, select_min_acer_threshold
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        selected = select_min_acer_threshold(probabilities, targets)["selected"]
        return {"acer": float(selected["acer"]), "apcer": float(selected["apcer"]),
                "bpcer": float(selected["bpcer"]), "threshold": float(selected["threshold"]),
                "nll": float(negative_log_likelihood(probabilities, targets)),
                "roc_auc": float(roc_auc(probabilities, targets)),
                "samples": int(targets.size)}

    def validate(self) -> dict[str, Any]:
        """The Table 37 checkpoint criterion, computed on `source_dev` only.

        Per-domain metrics are computed BESIDE the pooled ones rather than
        instead of them. §15.4's P3-ready tuple is an equal-weight mean over
        CASIA-dev and MSU-dev, and a pooled average is weighted by whichever
        domain happens to contribute more rows — a different quantity. The
        pooled numbers stay because P1/P2 select on their single domain, where
        the two coincide, and every inherited row keeps the metric keys its
        history was written with.
        """
        logits, targets = self.source_dev_logits()
        pooled = self._dev_metrics(logits, targets)
        metrics: dict[str, Any] = {f"source_dev/{name}": value for name, value in pooled.items()}
        per_domain: dict[str, dict[str, float]] = {}
        grouped = self.validation().domain_positions()
        if len(grouped) > 1:
            for domain, positions in grouped.items():
                domain_logits, domain_targets = self.source_dev_logits(positions)
                per_domain[domain] = self._dev_metrics(domain_logits, domain_targets)
        metrics["source_dev/per_domain"] = per_domain
        metrics["source_dev/decision_logit_name"] = self.decision_logit_name
        metrics["source_dev/decision_score_name"] = self.decision_score_name
        return metrics

    def is_better(self, metrics: dict[str, Any]) -> bool:
        """ACER primary, BPCER tie-break, then calibration NLL (Table 37). Never a
        target metric — no target metric exists in this process."""
        if not self.best_metrics: return True
        for name in (self.config.selection_metric, self.config.tie_break_metric, self.config.calibration_metric):
            candidate, incumbent = float(metrics[name]), float(self.best_metrics[name])
            if candidate < incumbent - 1e-12: return True
            if candidate > incumbent + 1e-12: return False
        return False

    # --- checkpointing ------------------------------------------------------
    def checkpoint_path(self, kind: str) -> Path:
        return self.run_root / "checkpoints" / f"{kind}.pt"

    def sampler_state(self) -> dict[str, Any]:
        return self.samplers[self.stage].state(epoch=self.epoch, step=self.step_in_epoch)

    def save(self, kind: str = "last") -> str:
        path = self.checkpoint_path(kind)
        sha = save_checkpoint(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                              scaler=self.scaler, epoch=self.epoch, global_step=self.global_step,
                              stage=self.stage, identity=self.identity, sampler_state=self.sampler_state(),
                              prototype_identity=self.prototype_identity, best_metrics=self.best_metrics,
                              stage_lineage=self.lineage.payload(), history=self.history,
                              resolved_config=self.config.resolved(), ema_enabled=self.config.ema_enabled)
        return sha

    def finish(self) -> dict[str, Any]:
        """Close the run once every declared stage is COMPLETED.

        Without this the run summary reports `RUNNING` forever, and the acceptance
        report would describe a finished run as still in flight. The transition is
        refused unless the lineage really covers the whole declared flow.
        """
        completed = [str(entry["stage"]) for entry in self.lineage.payload()
                     if entry.get("status") == "COMPLETED"]
        outstanding = [stage for stage in self.stages if stage not in completed]
        if outstanding:
            return {"status": self.status, "completed_stages": completed,
                    "outstanding_stages": outstanding, "closed": False}
        self.status = check_status_transition("RUNNING", "COMPLETED")
        self.write_run_json()
        return {"status": self.status, "completed_stages": completed,
                "outstanding_stages": [], "closed": True}

    def reconcile_lineage_from_disk(self) -> list[str]:
        """Adopt the on-disk stage markers into the restored lineage.

        A stage checkpoints `last` and only then writes its `output_hashes.json`, so
        a checkpoint taken during a stage records that stage as RUNNING even though
        it finished. The on-disk `output_hashes.json` is the authoritative
        completion marker; without this a resumed run would report a completed stage
        as still running.
        """
        reconciled: list[str] = []
        for entry in self.lineage.entries:
            stage = str(entry["stage"])
            path = self.run_root / "stages" / stage / "output_hashes.json"
            if entry.get("status") == "COMPLETED" or not path.is_file(): continue
            recorded = json.loads(path.read_text(encoding="utf-8"))
            entry["output_hashes"] = {key: value for key, value in recorded.items() if key != "stage"}
            entry["status"] = "COMPLETED"
            entry["reconciled_from_disk"] = True
            reconciled.append(stage)
        return reconciled

    def resume(self, kind: str = "last") -> dict[str, Any]:
        """Strict resume. Every scientific identity must match or this raises."""
        payload = load_checkpoint(self.checkpoint_path(kind), expected_identity=self.identity)
        restored = apply_checkpoint(payload, model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, scaler=self.scaler)
        self.epoch, self.global_step = restored["epoch"], restored["global_step"]
        self.step_in_epoch = int(restored["sampler_state"].get("step", 0))
        self.stage = restored["stage"]
        self.prototype_identity = restored["prototype_identity"]
        self.best_metrics = restored["best_metrics"]
        self.history = restored["history"]
        self.lineage = StageLineage.from_payload(restored["stage_lineage"], order=self.stages)
        restored["reconciled_stages"] = self.reconcile_lineage_from_disk()
        self.status = check_status_transition("INTERRUPTED", "RESUMING")
        self.status = check_status_transition(self.status, "RUNNING")
        return restored

    # --- smoke --------------------------------------------------------------
    def smoke(self, *, steps: int = 5, resume_steps: int = 6, stage: str = "G5") -> dict[str, Any]:
        """A real short run: train `steps`, checkpoint, resume, continue to
        `resume_steps`. Exercises the whole path on real samples."""
        started = time.time()
        self.enter_stage("G1")
        first = self.run_epoch("G1", max_steps=1)
        self.complete_stage("G1", {"epochs": 0, "steps": len(first), "global_step": self.global_step})
        g2 = self.run_g2_smoke()
        self.enter_stage("G5")
        # G5 uses its own (mixed) sampler, so its epoch plan list starts at 0.
        self.step_in_epoch = 0
        g5_before = max(0, int(steps) - self.global_step)
        records = self.run_epoch("G5", start_step=0, max_steps=g5_before)
        checkpoint_sha = self.save("last")
        before = {"epoch": self.epoch, "global_step": self.global_step,
                  "step_in_epoch": self.step_in_epoch, "stage": self.stage}
        restored = self.resume("last")
        after = self.run_epoch("G5", start_step=self.step_in_epoch,
                               max_steps=max(0, int(resume_steps) - self.global_step))
        resumed_sha = self.save("last")
        self.complete_stage("G5", {"steps": self.global_step, "checkpoint_sha256": resumed_sha})
        return {"schema_version": TRAINER_SCHEMA_VERSION, "run_id": self.config.run_id,
                "device": self.device, "amp": {"enabled": self.amp_enabled, "dtype": str(self.amp_dtype)},
                "identity": self.identity.payload(), "g2": g2,
                "steps_before_checkpoint": before["global_step"], "checkpoint_sha256": checkpoint_sha,
                "resumed_from": restored["global_step"], "steps_after_resume": self.global_step,
                "restarted_at_zero": restored["global_step"] == 0,
                "sampler_step_before": before["step_in_epoch"],
                "sampler_step_after_resume": int(restored["sampler_state"].get("step", -1)),
                "sampler_continued": int(restored["sampler_state"].get("step", -1)) == before["step_in_epoch"],
                "resumed_checkpoint_sha256": resumed_sha,
                "first_records": records[:2], "after_resume_records": after[:2],
                "all_losses_finite": all(math.isfinite(record["L_total"]) for record in records + after),
                "checkpoint_summary": checkpoint_summary(self.checkpoint_path("last")),
                "seconds": round(time.time() - started, 2)}

    def run_g2_smoke(self) -> dict[str, Any]:
        """G2 with a single manifold warm-up step instead of two full epochs."""
        self.enter_stage("G2")
        embeddings, valid, audit = self.collect_live_embeddings()
        state = initialize_prototypes(embeddings, valid, k=self.variant.prototype_k,
                                      epsilon=self.detector_config.covariance_epsilon,
                                      seed=self.config.prototype_seed,
                                      region_names=self.model.manifold.region_names)
        self.model.manifold.load_state(state)
        population = self.dataset.population_identity(self.dataset.live_positions())
        self.prototype_identity = state.identity(config_hash=self.config.hash(), population_identity=population)
        self.model.set_distance_scale(self._distance_scale(embeddings, valid), freeze=True)
        export = write_prototypes_npz(self.stage_root("G2") / "prototypes.npz", state,
                                      config_hash=self.config.hash(), population_identity=population,
                                      feature_identity=self.model.architecture_identity())
        records = self.run_epoch("G2", start_step=self.step_in_epoch, max_steps=1)
        output = {"prototype_identity_sha256": self.prototype_identity, "prototypes": export,
                  "population": audit, "steps": len(records), "global_step": self.global_step}
        self.complete_stage("G2", output)
        return output

    # --- reporting ----------------------------------------------------------
    def run_summary(self) -> dict[str, Any]:
        return {"schema_version": TRAINER_SCHEMA_VERSION, "run_id": self.config.run_id,
                "device": self.device, "status": self.status, "stage": self.stage,
                "epoch": self.epoch, "global_step": self.global_step,
                "identity": self.identity.payload(), "stage_lineage": self.lineage.payload(),
                "stage_lineage_identity": self.lineage.identity(),
                "prototype_identity_sha256": self.prototype_identity,
                "best_metrics": dict(self.best_metrics), "ema_enabled": self.config.ema_enabled,
                "seed": int(self.config.seed), "determinism": dict(self.determinism),
                "amp": {"enabled": self.amp_enabled, "dtype": str(self.amp_dtype)},
                "parameter_counts": self.model.parameter_counts(),
                "dataset": self.dataset.summary(), "config": self.config.resolved(),
                "variant": self.variant.payload(), "declared_stages": list(self.stages),
                "optimizer_groups": [{"name": str(group.get("name", index)),
                                      "parameters": sum(p.numel() for p in group["params"]),
                                      "lr": float(group["lr"])}
                                     for index, group in enumerate(self.optimizer.param_groups)],
                "git_commit": git_commit(), "target_test_opened": False}


def source_isolation_report(trainer: M9Trainer, *, source_dev_opened: bool) -> dict[str, Any]:
    """The evidence spec section 3 / Table 23 requires for the reference run."""
    return {"schema_version": TRAINER_SCHEMA_VERSION,
            "source_train_opened": True,
            "m8_validated_bank_opened": True,
            "m8_bank_id": trainer.dataset.bank.bank_id,
            "m8_bank_identity_sha256": trainer.dataset.bank.identity,
            "source_dev_opened": bool(source_dev_opened),
            "source_dev_produced_gradient": False,
            "target_test_opened": False, "target_labels_opened": False,
            "raw_dataset_opened": False, "m8_rejected_candidates_used": False,
            "m8_v1_v2_banks_used": False,
            "source_package_identity_sha256": trainer.dataset.package_identity}

def run_source_only_flow(trainer: M9Trainer, *, resume: bool = True) -> dict[str, Any]:
    """The declared source-only flow, G1 -> G2 -> G5 -> G6, run to closure.

    Lifted out of the Modal entrypoint because it is not deployment glue: it is
    the sequence a detector run IS, and C7's search trials and C8's matrix rows
    both need it. Two copies of "which stage comes next after a resume" would be
    two scientific authorities, and they would disagree the first time one was
    edited.

    Three properties the loop has to have, and each is a consequence of the
    lineage rather than of a flag:

    * **Resume executes only what is outstanding.** The stage to run next comes
      from `lineage.current` after the checkpoint is restored, so a run
      interrupted inside G5 continues in G5 and does not redo G1.
    * **A variant without a manifold never enters G2.** `variant.required_stages`
      decides the flow; `stage_for_epoch` already spends those two epochs in G1,
      so the schedule length is identical and no comparison is confounded by it.
    * **G6 calibrates the BEST checkpoint, on source_dev alone.** Not the last
      one: the selected checkpoint and the calibrated one must be the same
      model, or the operating threshold belongs to a model nobody ships.

    Returns the merged per-stage evidence, the run closure, the run summary and
    the trainer's own source-isolation report. Opens no target.
    """
    resumed: dict[str, Any] | None = None
    if resume and trainer.checkpoint_path("last").is_file():
        resumed = trainer.resume("last")

    executed: dict[str, Any] = {}
    stages = tuple(trainer.stages)
    if trainer.lineage.current is None and "G1" in stages:
        executed["G1"] = trainer.run_g1()
    if trainer.lineage.current == "G1" and "G2" in stages:
        executed["G2"] = trainer.run_g2()
    if trainer.lineage.current in ("G1", "G2") and "G5" in stages:
        executed["G5"] = trainer.run_g5()
    if trainer.lineage.current == "G5" and "G6" in stages:
        executed["G6"] = trainer.run_g6(checkpoint=trainer.checkpoint_path("best"))
    closure = trainer.finish()

    # A resumed run executes only the outstanding stages, so per-stage evidence
    # comes from the restored lineage and is then overlaid with whatever THIS
    # invocation produced. Merged per stage rather than replaced, so a stage this
    # invocation ran keeps the lineage's status instead of dropping it.
    merged: dict[str, Any] = {}
    for entry in trainer.lineage.payload():
        merged[str(entry["stage"])] = {"status": entry.get("status"),
                                       **(entry.get("output_hashes") or {})}
    for name, value in executed.items():
        merged[name] = {**merged.get(name, {}), **value}

    return {
        "schema_version": TRAINER_SCHEMA_VERSION,
        "resumed_from": resumed["global_step"] if resumed else None,
        "resumed_stage": resumed["stage"] if resumed else None,
        "stages_executed_here": sorted(executed),
        "stages": merged,
        "declared_stages": list(stages),
        "run_closure": closure,
        "run_summary": trainer.run_summary(),
        "source_isolation": source_isolation_report(trainer, source_dev_opened=True),
        "best_checkpoint": trainer.checkpoint_path("best").as_posix(),
        "last_checkpoint": trainer.checkpoint_path("last").as_posix(),
        "target_test_opened": False,
    }
