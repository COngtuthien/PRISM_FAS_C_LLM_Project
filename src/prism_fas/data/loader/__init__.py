"""M4 canonical package loader and deterministic balanced sampler."""

from .collate import collate_source_batch, collate_target_batch
from .config import INFERENCE_SPLIT, LOADER_SCHEMA_VERSION, LoaderConfig, TRAINING_SPLIT, VALIDATION_SPLIT, load_loader_config
from .contracts import CanonicalGeometry, CanonicalSourceSample, CanonicalTargetSample, SampleContractError, TargetIsolationViolation
from .loose_dataset import CanonicalPackageDataset, build_dataset
from .package_index import PackageContractError, PackageIndex, open_package, package_summary
from .sampler import BalancedDomainClassBatchSampler, SamplerConfigurationError, batch_fingerprint
from .shard_dataset import CanonicalShardDataset

__all__=["collate_source_batch","collate_target_batch","INFERENCE_SPLIT","LOADER_SCHEMA_VERSION","LoaderConfig",
         "TRAINING_SPLIT","VALIDATION_SPLIT","load_loader_config","CanonicalGeometry","CanonicalSourceSample",
         "CanonicalTargetSample","SampleContractError","TargetIsolationViolation","CanonicalPackageDataset",
         "build_dataset","PackageContractError","PackageIndex","open_package","package_summary",
         "BalancedDomainClassBatchSampler","SamplerConfigurationError","batch_fingerprint","CanonicalShardDataset"]
