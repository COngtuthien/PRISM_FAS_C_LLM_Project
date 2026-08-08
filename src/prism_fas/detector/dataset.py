"""The M9 real + synthetic training dataset.

Real samples come from the EXISTING M3/M4 canonical loader (`CanonicalPackageDataset`
over the validated `prism_data_v1_m3b` package). There is no second source
preprocessing path in M9: the same decode, the same [0,1] RGB, the same frozen
priors.

Synthetic samples come from the frozen, validated M8 v3 bank through
`detector.synthetic_bank.SyntheticBankReader`, which is fail-closed on the bank
identity and yields accepted rows only.

Three sample kinds, exactly as the training contract declares them:

    REAL LIVE       source_train, label live,  M3B region priors + visibility
    REAL SPOOF      source_train, label spoof, M3B region priors + visibility
    SYNTHETIC       accepted M8 v3 bank, DECLARED spoof label, q as a weight,
                    artifact map, exact mask -> attack region mask, recipe identity

`q` never becomes a label. `source_dev` is reachable only through the explicit
validation class, and `target_test` is not reachable at all.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
from prism_fas.data.loader.config import INFERENCE_SPLIT, TRAINING_SPLIT, VALIDATION_SPLIT, LoaderConfig
from .contracts import LIVE, REGION_COUNT, SPOOF, DetectorBatch, TargetIsolationViolation
from .region_cache import (PRIOR_STORAGE_SIZE, load_or_build_attack_mask_cache,
                           load_or_build_region_prior_cache)
from .synthetic_bank import SyntheticBankReader, SyntheticSample

DATASET_SCHEMA_VERSION = "m9-dataset-v1"
DOMAINS = ("casia_fasd", "msu_mfsd")
IMAGE_SIZE = 224


class DatasetError(RuntimeError):
    """The M9 dataset cannot be assembled as declared."""


def assert_source_only(split: str) -> str:
    """M9 never opens the target split, under any mode, from any call site."""
    if split == INFERENCE_SPLIT or "target" in split:
        raise TargetIsolationViolation(f"M9 must never open {split!r}; target data is an M10 stage")
    return split


@dataclass(frozen=True)
class TrainingItem:
    """One assembled sample, independent of which pool it came from."""
    sample_id: str
    kind: str                              # real_live | real_spoof | synthetic_spoof
    image: np.ndarray                      # [3,224,224] float32
    label: int                             # live=0, spoof=1
    dataset: str
    region_priors: np.ndarray              # [R,56,56] float32
    visibility: np.ndarray                 # [R] float32
    is_synthetic: bool
    artifact_map: np.ndarray | None = None       # [1,224,224] float32
    attack_region_mask: np.ndarray | None = None # [R] float32
    quality_weight: float | None = None          # q — a WEIGHT, never a label
    recipe_index: int = -1
    recipe_id: str = ""
    artifact_family: str = ""


class M9TrainingDataset:
    """Real `source_train` plus the accepted M8 v3 synthetic bank."""

    def __init__(self, package_root: Path, bank_root: Path, loader_config: LoaderConfig, *,
                 cache_root: Path, recipe_ids: Sequence[str] = (), backend: str = "loose",
                 progress: Callable[[int, int], None] | None = None,
                 variant: Any = None, bank_identity: str | None = None, bank_id: str | None = None):
        from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
        from .variant import ResolvedExperimentVariant
        assert_source_only(TRAINING_SPLIT)
        self.variant = (variant or ResolvedExperimentVariant.reference()).validate()
        self.real = CanonicalPackageDataset(Path(package_root), TRAINING_SPLIT, loader_config, mode="training")
        # A row that declares `recipe_conditioning: random_operators` opens the M10
        # random-operator bank instead of the M7-structured M8 v3 bank. Both are
        # opened fail-closed against their own pinned identity; neither is a fallback
        # for the other, and no threshold or acceptance rule is shared by accident.
        open_kwargs: dict[str, Any] = {"backend": backend}
        if bank_identity: open_kwargs["expected_identity"] = bank_identity
        if bank_id: open_kwargs["expected_bank_id"] = bank_id
        self.bank = SyntheticBankReader.open(Path(bank_root), **open_kwargs)
        self.package_root = Path(package_root)
        self.package_identity = self.real.index.content_identity
        if self.bank.lock["package_identity"] != self.package_identity:
            raise DatasetError("the synthetic bank was built against a different source package")
        self.recipe_index = {str(value): position for position, value in enumerate(recipe_ids)}
        rows = list(self.real.index.rows)
        self._real_position = {row["sample_id"]: position for position, row in enumerate(rows)}
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        self.prior_cache, self.prior_cache_state = load_or_build_region_prior_cache(
            Path(cache_root), self.package_root, rows, package_identity=self.package_identity,
            split=TRAINING_SPLIT, progress=progress)
        self.attack_cache, self.attack_cache_state = load_or_build_attack_mask_cache(
            Path(cache_root), self.bank, self.package_root,
            {row["sample_id"]: row for row in rows}, progress=progress)
        if self.prior_cache.sample_ids != tuple(row["sample_id"] for row in rows):
            raise DatasetError("the region prior cache does not align with the source_train row order")
        self.real_live, self.real_spoof = self._real_pools()
        self.synthetic_routes = self._synthetic_pools()

    # --- pools --------------------------------------------------------------
    def _real_pools(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        live: dict[str, list[int]] = {}
        spoof: dict[str, list[int]] = {}
        for position, row in enumerate(self.real.index.rows):
            bucket = live if row["label_live_spoof"] == "live" else spoof
            bucket.setdefault(row["dataset"], []).append(position)
        return live, spoof

    def _synthetic_pools(self) -> dict[str, list[int]]:
        """Accepted rows by route, restricted to the routes the variant declares.

        A03 `physics_only` / `gpat_only` RESTRICT the accepted M8 v3 rows. They do
        not regenerate a bank, do not touch a rejected candidate and do not move a
        quality threshold — the bank identity is unchanged in every row that uses
        synthetic data.
        """
        allowed = set(self.variant.synthetic_routes)
        # `synthetic: none` declares NO routes, so the allowed set is empty and every
        # pool is empty too. Writing this as `if allowed and route not in allowed`
        # short-circuits on the empty set and admits all 871 accepted rows into pools
        # that are then never sampled — harmless to training (the batch quota is 0)
        # but it made `run.json` report 871 synthetic samples for a row that used
        # none. A row reports the pool it actually declares.
        if not allowed: return {}
        pools: dict[str, list[int]] = {}
        for position, row in enumerate(self.bank.rows):
            route = str(row["route"])
            if route not in allowed: continue
            pools.setdefault(route, []).append(position)
        return {name: sorted(values) for name, values in sorted(pools.items())}

    def live_positions(self) -> list[int]:
        """Every `source_train` LIVE row, in stable sample-id order. This is the
        exact prototype-initialization population (spec section 9.3): 280 rows,
        CASIA 160 / MSU 120, with no spoof, synthetic, source_dev or target row."""
        return sorted(position for values in self.real_live.values() for position in values)

    def population_identity(self, positions: Sequence[int]) -> str:
        ids = [self.real.index.rows[position]["sample_id"] for position in sorted(positions)]
        return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()

    # --- items --------------------------------------------------------------
    def real_item(self, position: int) -> TrainingItem:
        sample = self.real[int(position)]
        label = LIVE if sample.label == "live" else SPOOF
        return TrainingItem(sample_id=sample.sample_id,
                            kind="real_live" if label == LIVE else "real_spoof",
                            image=np.asarray(sample.image, dtype=np.float32), label=label,
                            dataset=sample.dataset, region_priors=self.prior_cache.prior(int(position)),
                            visibility=self.prior_cache.visible(int(position)), is_synthetic=False)

    def synthetic_item(self, position: int) -> TrainingItem:
        sample: SyntheticSample = self.bank.sample(int(position))
        live_position = self._real_position.get(sample.live_target_sample_id)
        if live_position is None:
            raise DatasetError(f"{sample.synthetic_id}: its live target is not in source_train")
        mask = self.attack_cache.masks[self.attack_cache.position_of(sample.synthetic_id)]
        return TrainingItem(
            sample_id=sample.synthetic_id, kind="synthetic_spoof",
            image=np.asarray(sample.image, dtype=np.float32),
            # The synthetic class label is DECLARED by the training contract; it is
            # never derived from q, from the artifact map or from the recipe.
            label=SPOOF, dataset=sample.live_target_dataset,
            region_priors=self.prior_cache.prior(live_position),
            visibility=self.prior_cache.visible(live_position), is_synthetic=True,
            artifact_map=np.asarray(sample.artifact_map, dtype=np.float32),
            attack_region_mask=np.asarray(mask, dtype=np.float32),
            # `q` is loaded unchanged whatever the variant does with it: the M8 gate
            # and its acceptance are never rebuilt. `quality_weighting` decides only
            # how the loss WEIGHTS an accepted sample, and q never becomes a label.
            quality_weight=float(sample.quality_weight),
            # `recipe_conditioning: off` means the recipe identity does not reach the
            # loss graph at all — no prompt target and no artifact-family risk group.
            recipe_index=(int(self.recipe_index.get(sample.recipe_id, -1))
                          if self.variant.consumes_recipe_identity else -1),
            recipe_id=sample.recipe_id if self.variant.consumes_recipe_identity else "",
            artifact_family=(self.bank.primary_artifact_family(int(position))
                             if self.variant.consumes_recipe_identity else ""))

    def batch_from_plan(self, plan: Any) -> DetectorBatch:
        """Assemble one `BatchPlan` into a validated `DetectorBatch`."""
        items = ([self.real_item(position) for position in plan.real_live]
                 + [self.real_item(position) for position in plan.real_spoof]
                 + [self.synthetic_item(position) for position in plan.synthetic])
        return collate_items(items)

    def live_batch(self, positions: Sequence[int]) -> DetectorBatch:
        """A LIVE-only batch, used by prototype initialization. Raises if any
        position is not a live row, so the population cannot drift."""
        allowed = set(self.live_positions())
        illegal = [position for position in positions if int(position) not in allowed]
        if illegal: raise DatasetError(f"{len(illegal)} non-live rows were offered to a live-only batch")
        return collate_items([self.real_item(position) for position in positions])

    def summary(self) -> dict[str, Any]:
        return {"schema_version": DATASET_SCHEMA_VERSION,
                "package_identity_sha256": self.package_identity,
                "bank_id": self.bank.bank_id, "bank_identity_sha256": self.bank.identity,
                "real_total": len(self.real),
                "real_live": {name: len(values) for name, values in sorted(self.real_live.items())},
                "real_spoof": {name: len(values) for name, values in sorted(self.real_spoof.items())},
                "synthetic_routes": {name: len(values) for name, values in self.synthetic_routes.items()},
                "synthetic_total": sum(len(values) for values in self.synthetic_routes.values()),
                "synthetic_accepted_in_bank": len(self.bank),
                "declared_routes": list(self.variant.synthetic_routes),
                "recipe_source": self.variant.recipe_source,
                "consumes_recipe_identity": self.variant.consumes_recipe_identity,
                "recipe_count": len(self.recipe_index),
                "backend": self.bank.backend, "prior_storage_size": PRIOR_STORAGE_SIZE,
                "region_prior_cache": {"state": self.prior_cache_state,
                                       "identity_sha256": self.prior_cache.identity},
                "attack_mask_cache": {"state": self.attack_cache_state,
                                      "identity_sha256": self.attack_cache.identity},
                "live_population": len(self.live_positions()),
                "live_population_identity_sha256": self.population_identity(self.live_positions())}

    def contract_identity(self) -> str:
        return dataset_contract_identity({
            "package_identity": self.package_identity, "bank_identity": self.bank.identity,
            "region_prior_cache_identity": self.prior_cache.identity,
            "attack_mask_cache_identity": self.attack_cache.identity,
            "rows_identity": self.bank.rows_identity(), "recipe_count": len(self.recipe_index)})


class M9ValidationDataset:
    """`source_dev`, opened ONLY through this explicit constructor.

    Spec section 3 and section 10.3: `source_dev` is read for checkpoint selection
    and the source-only calibration and produces no optimization gradient. It is a
    separate class precisely so that reading it is never accidental.
    """

    def __init__(self, package_root: Path, loader_config: LoaderConfig, *, cache_root: Path,
                 limit: int | None = None, progress: Callable[[int, int], None] | None = None):
        from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
        assert_source_only(VALIDATION_SPLIT)
        self.real = CanonicalPackageDataset(Path(package_root), VALIDATION_SPLIT, loader_config, mode="validation")
        self.package_identity = self.real.index.content_identity
        rows = list(self.real.index.rows)
        self.positions = tuple(range(len(rows) if limit is None else min(int(limit), len(rows))))
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        self.prior_cache, self.prior_cache_state = load_or_build_region_prior_cache(
            Path(cache_root), Path(package_root), rows, package_identity=self.package_identity,
            split=VALIDATION_SPLIT, progress=progress)

    def __len__(self) -> int: return len(self.positions)

    def item(self, position: int) -> TrainingItem:
        sample = self.real[int(position)]
        label = LIVE if sample.label == "live" else SPOOF
        return TrainingItem(sample_id=sample.sample_id,
                            kind="real_live" if label == LIVE else "real_spoof",
                            image=np.asarray(sample.image, dtype=np.float32), label=label,
                            dataset=sample.dataset, region_priors=self.prior_cache.prior(int(position)),
                            visibility=self.prior_cache.visible(int(position)), is_synthetic=False)

    def batch(self, positions: Sequence[int]) -> DetectorBatch:
        return collate_items([self.item(position) for position in positions])

    def batches(self, batch_size: int = 32) -> Any:
        for start in range(0, len(self.positions), int(batch_size)):
            yield self.batch(self.positions[start:start + int(batch_size)])

    def summary(self) -> dict[str, Any]:
        return {"schema_version": DATASET_SCHEMA_VERSION, "split": VALIDATION_SPLIT,
                "package_identity_sha256": self.package_identity, "rows": len(self.positions),
                "available_rows": len(self.real), "produces_gradient": False,
                "region_prior_cache": {"state": self.prior_cache_state,
                                       "identity_sha256": self.prior_cache.identity}}


def collate_items(items: Sequence[TrainingItem]) -> DetectorBatch:
    """Stack items into the validated `DetectorBatch` of Table 52."""
    if not items: raise DatasetError("cannot collate an empty batch")
    datasets = tuple(sorted({item.dataset for item in items}))
    dataset_id = {name: index for index, name in enumerate(datasets)}
    stack = lambda values, dtype: torch.from_numpy(np.ascontiguousarray(np.stack(values)).astype(dtype))
    synthetic = np.array([item.is_synthetic for item in items], dtype=bool)
    batch = DetectorBatch(
        image=stack([item.image for item in items], np.float32),
        label=torch.tensor([item.label for item in items], dtype=torch.long),
        dataset_id=torch.tensor([dataset_id[item.dataset] for item in items], dtype=torch.long),
        is_synthetic=torch.from_numpy(synthetic),
        region_priors=stack([item.region_priors for item in items], np.float32),
        visibility=stack([item.visibility for item in items], np.float32),
        sample_ids=tuple(item.sample_id for item in items), datasets=datasets,
        artifact_map=(stack([item.artifact_map if item.artifact_map is not None
                             else np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                             for item in items], np.float32) if synthetic.any() else None),
        attack_region_mask=stack([item.attack_region_mask if item.attack_region_mask is not None
                                  else np.zeros(REGION_COUNT, dtype=np.float32) for item in items], np.float32),
        quality_weight=torch.tensor([float(item.quality_weight or 0.0) for item in items], dtype=torch.float32),
        recipe_index=torch.tensor([int(item.recipe_index) for item in items], dtype=torch.long),
        artifact_family=tuple(item.artifact_family for item in items))
    return batch.validate()


def batch_composition(batch: DetectorBatch) -> dict[str, int]:
    counts = {"real_live": 0, "real_spoof": 0, "synthetic_spoof": 0}
    for kind in batch.kinds(): counts[kind] += 1
    return counts


def domain_composition(batch: DetectorBatch) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for kind, index in zip(batch.kinds(), batch.dataset_id.tolist()):
        out.setdefault(kind, {}).setdefault(batch.datasets[index], 0)
        out[kind][batch.datasets[index]] += 1
    return {kind: dict(sorted(values.items())) for kind, values in sorted(out.items())}


def dataset_contract_identity(payload: dict[str, Any]) -> str:
    body = {"schema_version": DATASET_SCHEMA_VERSION, **payload}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()
