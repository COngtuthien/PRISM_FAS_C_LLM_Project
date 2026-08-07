"""M9 regional CNN-VLM detector, real manifolds, declared losses and stage trainer.

Implements `docs/M9_DETECTOR_CONTRACT.md` and `docs/M9_TRAINING_CONTRACT.md`, which
trace every requirement back to `docs/spec_snapshot.md`.

Source-only: this package reads the frozen `prism_data_v1_m3b` source splits and the
frozen, validated M8 synthetic bank. It never opens `target_test`, a target label, a
raw dataset, a rejected M8 candidate or a v1/v2 bank. It must never import modal
(spec section 12.1).
"""
from .contracts import (LIVE, M9_SCHEMA_VERSION, REGION_COUNT, REGION_ORDER, SPOOF,
                        DetectorBatch, DetectorContractError, ModelOutput,
                        TargetIsolationViolation, architecture_identity)
from .manifold import (DEFAULT_COVARIANCE_EPSILON, DEFAULT_K, DEFAULT_TAU_PROTOTYPE,
                       ManifoldError, PrototypeState, RealManifold, deterministic_kmeans,
                       initialize_prototypes, read_prototypes_npz, write_prototypes_npz)
from .losses import (DEFAULT_WEIGHTS, LOSS_NAMES, LossError, LossResult, compute_losses,
                     loss_contract_identity)
from .pretrained import (CONVNEXT_PIN, SIGLIP2_PIN, SIGLIP2_TOKENIZATION, PretrainedError,
                         SigLIP2Artifacts, pretrained_manifest, resolve_convnext_weight)
from .heads import (GlobalHead, PromptHead, RecipeTextCache, TextCacheError,
                    build_recipe_text_cache, read_recipe_text_cache, write_recipe_text_cache)
from .prism_detector import DetectorConfig, PRISMDetector, build_detector, detector_summary
from .synthetic_bank import (FROZEN_BANK_ID, FROZEN_BANK_IDENTITY, SyntheticBankAccessError,
                             SyntheticBankReader, SyntheticSample, parity_audit)
from .sampler import BatchContract, BatchPlan, M9BatchSampler, SamplerError
from .dataset import (DatasetError, M9TrainingDataset, M9ValidationDataset, TrainingItem,
                      batch_composition, collate_items, domain_composition)
from .checkpoint import (M9_CHECKPOINT_SCHEMA_VERSION, STAGE_ORDER, M9CheckpointError,
                         RunIdentity, StageLineage, StageTransitionError, apply_checkpoint,
                         check_stage_transition, check_status_transition, checkpoint_summary,
                         load_checkpoint, save_checkpoint)
from .trainer import M9Trainer, M9TrainingConfig, TrainerError, enabled_terms, source_isolation_report

__all__ = [
    "LIVE", "SPOOF", "M9_SCHEMA_VERSION", "REGION_ORDER", "REGION_COUNT",
    "DetectorBatch", "ModelOutput", "DetectorContractError", "TargetIsolationViolation",
    "architecture_identity", "RealManifold", "PrototypeState", "ManifoldError",
    "deterministic_kmeans", "initialize_prototypes", "write_prototypes_npz",
    "read_prototypes_npz", "DEFAULT_K", "DEFAULT_COVARIANCE_EPSILON", "DEFAULT_TAU_PROTOTYPE",
    "compute_losses", "LossResult", "LossError", "LOSS_NAMES", "DEFAULT_WEIGHTS",
    "loss_contract_identity", "SIGLIP2_PIN", "SIGLIP2_TOKENIZATION", "CONVNEXT_PIN",
    "SigLIP2Artifacts", "PretrainedError", "resolve_convnext_weight", "pretrained_manifest",
    "GlobalHead", "PromptHead", "RecipeTextCache", "TextCacheError", "build_recipe_text_cache",
    "read_recipe_text_cache", "write_recipe_text_cache", "PRISMDetector", "DetectorConfig",
    "build_detector", "detector_summary", "SyntheticBankReader", "SyntheticSample",
    "SyntheticBankAccessError", "FROZEN_BANK_ID", "FROZEN_BANK_IDENTITY", "parity_audit",
    "BatchContract", "BatchPlan", "M9BatchSampler", "SamplerError", "M9TrainingDataset",
    "M9ValidationDataset", "TrainingItem", "DatasetError", "collate_items", "batch_composition",
    "domain_composition", "save_checkpoint", "load_checkpoint", "apply_checkpoint",
    "checkpoint_summary", "RunIdentity", "StageLineage", "M9CheckpointError",
    "StageTransitionError", "check_stage_transition", "check_status_transition",
    "STAGE_ORDER", "M9_CHECKPOINT_SCHEMA_VERSION", "M9Trainer", "M9TrainingConfig",
    "TrainerError", "enabled_terms", "source_isolation_report"]
