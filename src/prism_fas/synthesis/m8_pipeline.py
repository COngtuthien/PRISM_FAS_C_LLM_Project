from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from prism_fas.data.loader.config import load_loader_config
from prism_fas.data.loader.transforms import read_image
from prism_fas.data.package.priors import load_prior
from prism_fas.recipes.bank import load_bank
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.conditioning import conditioning_vector
from .gpat_contracts import GPATBatch
from .masks import RegionMaskBuilder
from .pair_plan import SOURCE_SPLIT, load_pair_manifest

M8_PIPELINE_SCHEMA_VERSION = "m8-pipeline-v1"
PROJECT_ROOT = Path(__file__).parents[3]
LOADER_CONFIG = PROJECT_ROOT / "configs" / "data" / "loader_m4.yaml"
FORBIDDEN_SPLITS = ("source_dev", "target_test")
ALLOWED_DATASETS = ("casia_fasd", "msu_mfsd")


class PipelineError(RuntimeError):
    """An M8 pipeline stage was asked for data it is not allowed to open."""


class SourceOnlyAudit:
    """Records every package path M8 opens so isolation can be proven, not claimed."""
    def __init__(self) -> None:
        self.opened: list[str] = []
    def record(self, relative_path: str) -> str:
        text = str(relative_path).replace("\\", "/")
        lowered = text.lower()
        for token in FORBIDDEN_SPLITS:
            if token in lowered: raise PipelineError(f"M8 refused to open a forbidden artifact: {text}")
        if "siw" in lowered or "target" in lowered: raise PipelineError(f"M8 refused to open target data: {text}")
        self.opened.append(text)
        return text
    def report(self) -> dict[str, Any]:
        manifests = sorted({path for path in self.opened if path.startswith("manifests/")})
        return {"source_train_opened": True, "source_dev_opened": False, "target_test_opened": False,
                "target_label_artifact_opened": False, "raw_dataset_path_opened": False,
                "manifests_opened": manifests, "distinct_paths": len(set(self.opened)),
                "total_opens": len(self.opened),
                "path_prefixes": sorted({path.split("/", 1)[0] for path in self.opened})}


def load_gpat_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise PipelineError("gpat config must be a YAML mapping")
    data = payload.get("data", {})
    if data.get("package_split") != SOURCE_SPLIT:
        raise PipelineError(f"gpat config package_split must be {SOURCE_SPLIT!r}")
    for dataset in data.get("allowed_datasets", []):
        if dataset not in ALLOWED_DATASETS: raise PipelineError(f"gpat config allows dataset {dataset!r}")
    text = json.dumps(payload, sort_keys=True)
    for token in FORBIDDEN_SPLITS + ("siw", "target_test"):
        if token in text.lower() and token not in json.dumps(data.get("forbidden_splits", []), sort_keys=True).lower():
            raise PipelineError(f"gpat config references {token!r} outside its forbidden list")
    return payload


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class SampleStore:
    """Reads source_train image/prior payloads only, with an open-audit trail."""
    package_root: Path
    audit: SourceOnlyAudit
    _rows: dict[str, dict[str, Any]]
    _cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]

    @classmethod
    def open(cls, package_root: Path, audit: SourceOnlyAudit | None = None) -> "SampleStore":
        import pyarrow.parquet as pq
        audit = audit or SourceOnlyAudit()
        root = Path(package_root)
        relative = f"manifests/{SOURCE_SPLIT}.parquet"
        audit.record(relative)
        table = pq.read_table(root / relative).to_pydict()
        rows: dict[str, dict[str, Any]] = {}
        for index in range(len(table["sample_id"])):
            if table["project_split"][index] != SOURCE_SPLIT:
                raise PipelineError(f"row {index} is not {SOURCE_SPLIT}")
            rows[table["sample_id"][index]] = {key: table[key][index] for key in table}
        return cls(package_root=root, audit=audit, _rows=rows, _cache={})

    def row(self, sample_id: str) -> dict[str, Any]:
        try: return self._rows[sample_id]
        except KeyError: raise PipelineError(f"sample {sample_id} is not in {SOURCE_SPLIT}") from None

    def load(self, sample_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if sample_id in self._cache: return self._cache[sample_id]
        row = self.row(sample_id)
        loader = load_loader_config(LOADER_CONFIG)
        image = read_image(self.package_root / self.audit.record(row["image_relative_path"]), loader.image)
        arrays = load_prior(self.package_root / self.audit.record(row["prior_relative_path"]))
        self._cache[sample_id] = (image, arrays)
        return image, arrays

    def mask_builder(self, sample_id: str) -> RegionMaskBuilder:
        _, arrays = self.load(sample_id)
        return RegionMaskBuilder(height=224, width=224, parsing=arrays["parsing_labels"],
                                 landmarks=arrays["landmarks"], bbox=arrays["bbox"], crop_box=arrays["crop_box"])


def build_batch(store: SampleStore, pairs: list[dict[str, Any]], bank: dict[str, Any],
                identity_model: Any, *, device: str = "cpu") -> GPATBatch:
    """Materialize one GPAT batch from real source_train pairs.

    The live embedding is produced by the same differentiable wrapper used for
    the generated embedding, then detached and cached, so the identity cosine is
    self-consistent.
    """
    recipes = {recipe.recipe_id: recipe for recipe in bank["recipes"]}
    live_images, spoof_images, conditionings, supports, styles, strengths = [], [], [], [], [], []
    for pair in pairs:
        recipe = recipes[pair["recipe_id"]]
        graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        live_image, _ = store.load(pair["live_sample_id"])
        spoof_image, _ = store.load(pair["spoof_sample_id"])
        policy = graph.region_mask_policy
        live_masks = store.mask_builder(pair["live_sample_id"]).build(
            list(graph.requested_regions), geometry_shape=str(policy["geometry_shape"]),
            coverage=float(policy["requested_coverage"]),
            seed=graph.node_seed(graph.nodes[0], f"{pair['live_sample_id']}|region_mask"))
        spoof_masks = store.mask_builder(pair["spoof_sample_id"]).build(
            list(graph.requested_regions), geometry_shape=str(policy["geometry_shape"]), coverage=1.0,
            seed=graph.node_seed(graph.nodes[0], f"{pair['spoof_sample_id']}|style_mask"))
        live_images.append(live_image)
        spoof_images.append(spoof_image)
        conditionings.append(conditioning_vector(recipe, bank["ontology"]))
        supports.append(np.asarray(live_masks.operator_support_mask, dtype=np.float32))
        styles.append(np.asarray(spoof_masks.requested_region_mask, dtype=np.float32))
        strengths.append(float(np.mean([spec.strength for spec in recipe.artifacts])))
    live = torch.from_numpy(np.stack(live_images)).to(device)
    spoof = torch.from_numpy(np.stack(spoof_images)).to(device)
    with torch.no_grad():
        live_embedding = identity_model(live).detach()
    return GPATBatch(live_image=live, source_spoof_image=spoof,
                     recipe_conditioning=torch.from_numpy(np.stack(conditionings)).to(device),
                     target_support_mask=torch.from_numpy(np.stack(supports)).to(device),
                     source_style_mask=torch.from_numpy(np.stack(styles)).to(device),
                     recipe_strength=torch.tensor(strengths, dtype=torch.float32, device=device),
                     live_identity_embedding=live_embedding,
                     pair_ids=tuple(pair["pair_id"] for pair in pairs),
                     live_sample_ids=tuple(pair["live_sample_id"] for pair in pairs),
                     spoof_sample_ids=tuple(pair["spoof_sample_id"] for pair in pairs),
                     recipe_ids=tuple(pair["recipe_id"] for pair in pairs)).validate()


def load_pairs(pairs_root: Path, partition: str) -> list[dict[str, Any]]:
    name = {"train": "pair_manifest_train.parquet", "validation": "pair_manifest_validation.parquet"}[partition]
    return load_pair_manifest(Path(pairs_root) / name)


def resolve_bank(bank_root: Path) -> dict[str, Any]:
    return load_bank(bank_root)
