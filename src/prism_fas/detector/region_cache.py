"""Precomputed, deterministic region priors and regional attack masks.

Building one sample's nine soft priors through the exact M7 `RegionMaskBuilder`
costs ~65 ms, which would dominate a 45-step epoch. A prior is a pure function of
the frozen package geometry, so it is computed once and reused; caching can change
timing but never values.

Storage resolution: the detector consumes priors only on the 14x14 SigLIP2 patch
grid and the 7x7 ConvNeXt stage-4 grid. 224 -> 56 -> 14 and 224 -> 56 -> 7 are exact
integer area reductions (4x then 4x = 16x = 224/14; 4x then 8x = 32x = 224/7), so
storing at 56x56 is not an approximation of the 224 prior on those grids.

The regional attack mask `m_r` needs the FULL-resolution prior against the M8 exact
edit mask, so it is computed at 224 and stored as the resulting `[M,9]` matrix.

Never imports modal. Reads only the frozen package and the frozen bank.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
from .contracts import REGION_COUNT, REGION_ORDER
from .npz_io import read_arrays_npz, write_arrays_npz
from .regions import PRIOR_BLUR_SIGMA, attack_region_mask, build_region_priors

REGION_CACHE_SCHEMA_VERSION = "m9-region-cache-v1"
PRIOR_STORAGE_SIZE = 56
ATTACK_OVERLAP = 0.05


class RegionCacheError(RuntimeError):
    """The region prior cache is missing, malformed or bound to other inputs."""


def _downsample(priors: np.ndarray, size: int = PRIOR_STORAGE_SIZE) -> np.ndarray:
    """Exact integer-block area reduction `[R,224,224] -> [R,size,size]`."""
    regions, height, width = priors.shape
    if height % size or width % size:
        raise RegionCacheError(f"{height}x{width} does not reduce exactly to {size}x{size}")
    factor_y, factor_x = height // size, width // size
    return priors.reshape(regions, size, factor_y, size, factor_x).mean(axis=(2, 4)).astype(np.float32)


@dataclass(frozen=True)
class RegionPriorCache:
    """Priors and visibility for one split, in the split's own sample-id order."""
    sample_ids: tuple[str, ...]
    priors: np.ndarray                 # [N,R,56,56] float16
    visibility: np.ndarray             # [N,R] float32
    binding: dict[str, Any]
    identity: str

    def validate(self) -> "RegionPriorCache":
        count = len(self.sample_ids)
        if self.priors.shape != (count, REGION_COUNT, PRIOR_STORAGE_SIZE, PRIOR_STORAGE_SIZE):
            raise RegionCacheError(f"priors must be [{count},{REGION_COUNT},{PRIOR_STORAGE_SIZE},{PRIOR_STORAGE_SIZE}]")
        if self.visibility.shape != (count, REGION_COUNT):
            raise RegionCacheError(f"visibility must be [{count},{REGION_COUNT}]")
        values = self.priors.astype(np.float32)
        if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0 + 1e-3:
            raise RegionCacheError("cached priors are not finite in [0,1]")
        if not np.isfinite(self.visibility).all():
            raise RegionCacheError("cached visibility is not finite")
        return self

    def position_of(self, sample_id: str) -> int:
        cached = getattr(self, "_lookup", None)
        if cached is None:
            cached = {value: index for index, value in enumerate(self.sample_ids)}
            object.__setattr__(self, "_lookup", cached)
        if sample_id not in cached: raise KeyError(f"{sample_id!r} is not in the region prior cache")
        return cached[sample_id]

    def prior(self, position: int) -> np.ndarray:
        return np.ascontiguousarray(self.priors[int(position)].astype(np.float32))

    def visible(self, position: int) -> np.ndarray:
        return np.ascontiguousarray(self.visibility[int(position)].astype(np.float32))


def cache_binding(*, package_identity: str, split: str, sample_ids: Sequence[str],
                  content_identity: str) -> dict[str, Any]:
    """Path-free, machine-free, timestamp-free binding."""
    return {"schema_version": REGION_CACHE_SCHEMA_VERSION, "package_identity_sha256": package_identity,
            "split": split, "region_order": list(REGION_ORDER), "storage_size": PRIOR_STORAGE_SIZE,
            "blur_sigma": float(PRIOR_BLUR_SIGMA), "mask_builder": "prism_fas.synthesis.masks.RegionMaskBuilder",
            "sample_count": len(sample_ids),
            "sample_ids_sha256": hashlib.sha256("|".join(sample_ids).encode("utf-8")).hexdigest(),
            "content_identity_sha256": content_identity}


def _identity(binding: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_region_prior_cache(package_root: Path, rows: Sequence[dict[str, Any]], *,
                             package_identity: str, split: str,
                             progress: Callable[[int, int], None] | None = None) -> RegionPriorCache:
    """Build the nine soft priors for every row, in the given (stable) row order."""
    from prism_fas.data.package.priors import load_prior
    root = Path(package_root)
    count = len(rows)
    priors = np.zeros((count, REGION_COUNT, PRIOR_STORAGE_SIZE, PRIOR_STORAGE_SIZE), dtype=np.float16)
    visibility = np.zeros((count, REGION_COUNT), dtype=np.float32)
    for position, row in enumerate(rows):
        arrays = load_prior(root / row["prior_relative_path"])
        full = build_region_priors(arrays["parsing_labels"], arrays["landmarks"],
                                   arrays["bbox"], arrays["crop_box"])
        priors[position] = _downsample(full).astype(np.float16)
        visibility[position] = np.asarray(arrays["visibility"], dtype=np.float32)
        if progress and (position + 1) % 100 == 0: progress(position + 1, count)
    ids = tuple(str(row["sample_id"]) for row in rows)
    content = hashlib.sha256(np.ascontiguousarray(priors).tobytes()
                             + np.ascontiguousarray(visibility).tobytes()).hexdigest()
    binding = cache_binding(package_identity=package_identity, split=split, sample_ids=ids,
                            content_identity=content)
    return RegionPriorCache(sample_ids=ids, priors=priors, visibility=visibility,
                            binding=binding, identity=_identity(binding)).validate()


def write_region_prior_cache(path: Path, cache: RegionPriorCache) -> dict[str, Any]:
    cache.validate()
    written = write_arrays_npz(Path(path), {
        "sample_ids": np.array(list(cache.sample_ids), dtype="U64"),
        "priors": cache.priors.astype(np.float16), "visibility": cache.visibility.astype(np.float32),
        "identity": np.array([cache.identity], dtype="U64"),
        "binding_json": np.frombuffer(json.dumps(cache.binding, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8"), dtype=np.uint8)})
    return {**written, "region_prior_cache_identity_sha256": cache.identity, "samples": len(cache.sample_ids)}


def read_region_prior_cache(path: Path, *, expected_identity: str | None = None) -> RegionPriorCache:
    arrays = read_arrays_npz(Path(path))
    binding = json.loads(bytes(arrays["binding_json"].astype(np.uint8)).decode("utf-8"))
    if binding.get("schema_version") != REGION_CACHE_SCHEMA_VERSION:
        raise RegionCacheError(f"region cache schema {binding.get('schema_version')!r} is not "
                               f"{REGION_CACHE_SCHEMA_VERSION!r}")
    cache = RegionPriorCache(sample_ids=tuple(str(value) for value in arrays["sample_ids"]),
                             priors=np.asarray(arrays["priors"], dtype=np.float16),
                             visibility=np.asarray(arrays["visibility"], dtype=np.float32),
                             binding=binding, identity=str(arrays["identity"][0])).validate()
    if _identity(binding) != cache.identity:
        raise RegionCacheError("stored region cache identity does not match its own binding")
    content = hashlib.sha256(np.ascontiguousarray(cache.priors).tobytes()
                             + np.ascontiguousarray(cache.visibility).tobytes()).hexdigest()
    if binding.get("content_identity_sha256") != content:
        raise RegionCacheError("stored region cache arrays do not match their content identity")
    if expected_identity and cache.identity != expected_identity:
        raise RegionCacheError(f"region cache identity {cache.identity} != expected {expected_identity}")
    return cache


def load_or_build_region_prior_cache(cache_root: Path, package_root: Path, rows: Sequence[dict[str, Any]], *,
                                     package_identity: str, split: str,
                                     progress: Callable[[int, int], None] | None = None) -> tuple[RegionPriorCache, str]:
    """Reuse a matching cache or build it. The file name carries the split and the
    package identity prefix, so a different package can never silently hit."""
    target = Path(cache_root) / f"m9_region_priors_{split}_{package_identity[:12]}.npz"
    if target.is_file():
        try: return read_region_prior_cache(target), "reused"
        except (RegionCacheError, KeyError, ValueError): pass
    cache = build_region_prior_cache(package_root, rows, package_identity=package_identity,
                                     split=split, progress=progress)
    write_region_prior_cache(target, cache)
    return cache, "built"


# --- regional attack masks --------------------------------------------------

@dataclass(frozen=True)
class AttackMaskCache:
    """`m_r` per accepted synthetic sample, from the M8 EXACT edit mask."""
    synthetic_ids: tuple[str, ...]
    masks: np.ndarray                  # [M,R] float32 in {0,1}
    binding: dict[str, Any]
    identity: str

    def validate(self) -> "AttackMaskCache":
        if self.masks.shape != (len(self.synthetic_ids), REGION_COUNT):
            raise RegionCacheError(f"attack masks must be [{len(self.synthetic_ids)},{REGION_COUNT}]")
        if set(np.unique(self.masks).tolist()) - {0.0, 1.0}:
            raise RegionCacheError("an attack mask holds a value other than 0 or 1")
        return self

    def position_of(self, synthetic_id: str) -> int:
        cached = getattr(self, "_lookup", None)
        if cached is None:
            cached = {value: index for index, value in enumerate(self.synthetic_ids)}
            object.__setattr__(self, "_lookup", cached)
        if synthetic_id not in cached: raise KeyError(f"{synthetic_id!r} is not in the attack mask cache")
        return cached[synthetic_id]


def build_attack_mask_cache(bank: Any, package_root: Path, rows_by_sample: dict[str, dict[str, Any]], *,
                            progress: Callable[[int, int], None] | None = None) -> AttackMaskCache:
    """`m_r = 1` where the M8 exact edit mask covers at least `ATTACK_OVERLAP` of the
    region's FULL-resolution prior mass.

    Derived from the frozen exact mask alone — never from `q`, never from a label.
    """
    from prism_fas.data.package.priors import load_prior
    root = Path(package_root)
    full_priors: dict[str, np.ndarray] = {}
    ids: list[str] = []
    masks = np.zeros((len(bank), REGION_COUNT), dtype=np.float32)
    for position in range(len(bank)):
        sample = bank.sample(position)
        live_id = sample.live_target_sample_id
        prior = full_priors.get(live_id)
        if prior is None:
            row = rows_by_sample.get(live_id)
            if row is None: raise RegionCacheError(f"{sample.synthetic_id}: live target is not in source_train")
            arrays = load_prior(root / row["prior_relative_path"])
            prior = build_region_priors(arrays["parsing_labels"], arrays["landmarks"],
                                        arrays["bbox"], arrays["crop_box"])
            full_priors[live_id] = prior
        masks[position] = attack_region_mask(sample.exact_mask, prior, overlap=ATTACK_OVERLAP)
        ids.append(sample.synthetic_id)
        if progress and (position + 1) % 100 == 0: progress(position + 1, len(bank))
    binding = {"schema_version": REGION_CACHE_SCHEMA_VERSION, "kind": "attack_masks",
               "bank_id": bank.bank_id, "bank_identity_sha256": bank.identity,
               "rows_identity_sha256": bank.rows_identity(), "overlap": float(ATTACK_OVERLAP),
               "region_order": list(REGION_ORDER), "blur_sigma": float(PRIOR_BLUR_SIGMA),
               "sample_count": len(ids),
               "content_identity_sha256": hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest()}
    return AttackMaskCache(synthetic_ids=tuple(ids), masks=masks, binding=binding,
                           identity=_identity(binding)).validate()


def write_attack_mask_cache(path: Path, cache: AttackMaskCache) -> dict[str, Any]:
    cache.validate()
    written = write_arrays_npz(Path(path), {
        "synthetic_ids": np.array(list(cache.synthetic_ids), dtype="U64"),
        "masks": cache.masks.astype(np.float32),
        "identity": np.array([cache.identity], dtype="U64"),
        "binding_json": np.frombuffer(json.dumps(cache.binding, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8"), dtype=np.uint8)})
    return {**written, "attack_mask_cache_identity_sha256": cache.identity, "samples": len(cache.synthetic_ids)}


def read_attack_mask_cache(path: Path, *, expected_identity: str | None = None) -> AttackMaskCache:
    arrays = read_arrays_npz(Path(path))
    binding = json.loads(bytes(arrays["binding_json"].astype(np.uint8)).decode("utf-8"))
    cache = AttackMaskCache(synthetic_ids=tuple(str(value) for value in arrays["synthetic_ids"]),
                            masks=np.asarray(arrays["masks"], dtype=np.float32), binding=binding,
                            identity=str(arrays["identity"][0])).validate()
    if _identity(binding) != cache.identity:
        raise RegionCacheError("stored attack mask identity does not match its own binding")
    if binding.get("content_identity_sha256") != hashlib.sha256(np.ascontiguousarray(cache.masks).tobytes()).hexdigest():
        raise RegionCacheError("stored attack masks do not match their content identity")
    if expected_identity and cache.identity != expected_identity:
        raise RegionCacheError(f"attack mask identity {cache.identity} != expected {expected_identity}")
    return cache


def load_or_build_attack_mask_cache(cache_root: Path, bank: Any, package_root: Path,
                                    rows_by_sample: dict[str, dict[str, Any]], *,
                                    progress: Callable[[int, int], None] | None = None) -> tuple[AttackMaskCache, str]:
    target = Path(cache_root) / f"m9_attack_masks_{bank.identity[:12]}.npz"
    if target.is_file():
        try: return read_attack_mask_cache(target), "reused"
        except (RegionCacheError, KeyError, ValueError): pass
    cache = build_attack_mask_cache(bank, package_root, rows_by_sample, progress=progress)
    write_attack_mask_cache(target, cache)
    return cache, "built"
