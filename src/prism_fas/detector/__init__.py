"""M9 regional CNN-VLM detector, real manifolds and declared losses.

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

__all__ = [
    "LIVE", "SPOOF", "M9_SCHEMA_VERSION", "REGION_ORDER", "REGION_COUNT",
    "DetectorBatch", "ModelOutput", "DetectorContractError", "TargetIsolationViolation",
    "architecture_identity", "RealManifold", "PrototypeState", "ManifoldError",
    "deterministic_kmeans", "initialize_prototypes", "write_prototypes_npz",
    "read_prototypes_npz", "DEFAULT_K", "DEFAULT_COVARIANCE_EPSILON", "DEFAULT_TAU_PROTOTYPE",
    "compute_losses", "LossResult", "LossError", "LOSS_NAMES", "DEFAULT_WEIGHTS",
    "loss_contract_identity"]
