"""M3A package foundation: deterministic priors, manifests, shards and lock."""

from .builder import build_package, build_priors, finalize_lock, load_m2_samples
from .config import M3APackageConfig, PACKAGE_SCHEMA_VERSION, PRIOR_SCHEMA_VERSION, QUALITY_SCHEMA_VERSION, load_package_config, project_split
from .quality import QUALITY_NAMES, QualityMetricError, compute_quality
from .selection import TargetIsolationError, available_splits, select_split_manifest
from .validator import validate_package, validate_source_m2_hashes

__all__=["build_package","build_priors","finalize_lock","load_m2_samples","M3APackageConfig","PACKAGE_SCHEMA_VERSION",
         "PRIOR_SCHEMA_VERSION","QUALITY_SCHEMA_VERSION","load_package_config","project_split","QUALITY_NAMES",
         "QualityMetricError","compute_quality","TargetIsolationError","available_splits","select_split_manifest",
         "validate_package","validate_source_m2_hashes"]
