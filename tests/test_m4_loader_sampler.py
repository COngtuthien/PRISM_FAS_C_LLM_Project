import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from prism_fas.data.loader import (BalancedDomainClassBatchSampler, CanonicalPackageDataset, CanonicalShardDataset,
                                   CanonicalSourceSample, CanonicalTargetSample, PackageContractError,
                                   SampleContractError, TargetIsolationViolation, collate_source_batch,
                                   collate_target_batch, load_loader_config, open_package)
from prism_fas.data.loader.contracts import FORBIDDEN_TARGET_FIELDS
from prism_fas.data.loader.sampler import SamplerConfigurationError, batch_fingerprint

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "data" / "loader_m4.yaml"
CONFIG = load_loader_config(CONFIG_PATH)
PACKAGE = Path(__file__).parents[1] / "data" / "processed" / "prism_data_v1_m3b"
package_required = pytest.mark.skipif(not (PACKAGE / "PACKAGE_LOCK.json").is_file(), reason="M3B package not built")
PRIVATE_TOKENS = ("live", "spoof", "attack", "taxonomy", "subject", "session", ".mov")


# --- config / package guards ----------------------------------------------

def test_label_mapping_is_explicit_and_committed():
    assert CONFIG.label_mapping == {"live": 0, "spoof": 1}
    assert CONFIG.label_to_index("spoof") == 1
    with pytest.raises(ValueError, match="outside the configured vocabulary"):
        CONFIG.label_to_index("unknown")
    assert "D:" not in CONFIG_PATH.read_text(encoding="utf-8")


@package_required
def test_package_identity_and_status_guards():
    index = open_package(PACKAGE, "source_train", CONFIG, mode="training")
    assert index.package_id == "prism_data_v1_m3b" and index.parent_package_id == "prism_data_v1_m3a"
    assert len(index) == 1440
    pinned = CONFIG.model_copy(update={"package": CONFIG.package.model_copy(update={"expected_content_identity_sha256": "f" * 64})})
    with pytest.raises(PackageContractError, match="content identity"):
        open_package(PACKAGE, "source_train", pinned, mode="training")
    wrong = CONFIG.model_copy(update={"package": CONFIG.package.model_copy(update={"expected_package_id": "other"})})
    with pytest.raises(PackageContractError, match="package id"):
        open_package(PACKAGE, "source_train", wrong, mode="training")


@package_required
def test_training_mode_rejects_target_and_dev_splits():
    with pytest.raises(TargetIsolationViolation, match="target isolation violation"):
        open_package(PACKAGE, "target_test", CONFIG, mode="training")
    with pytest.raises(PackageContractError, match="not permitted"):
        open_package(PACKAGE, "source_dev", CONFIG, mode="training")
    with pytest.raises(TargetIsolationViolation):
        BalancedDomainClassBatchSampler(open_package(PACKAGE, "source_dev", CONFIG, mode="validation"), CONFIG)


# --- sample contracts ------------------------------------------------------

@package_required
def test_source_sample_contract():
    dataset = CanonicalPackageDataset(PACKAGE, "source_train", CONFIG, mode="training")
    sample = dataset[0]
    assert isinstance(sample, CanonicalSourceSample)
    assert sample.image.shape == (3, 224, 224) and sample.image.dtype == np.float32
    assert 0.0 <= float(sample.image.min()) and float(sample.image.max()) <= 1.0
    assert sample.label in CONFIG.label_mapping and sample.class_target == CONFIG.label_mapping[sample.label]
    geometry = sample.geometry
    assert geometry.bbox.shape == (4,) and geometry.landmarks.shape == (5, 2)
    assert geometry.parsing_labels.shape == (224, 224) and geometry.parsing_labels.dtype == np.uint8
    assert geometry.pose_ypr.shape == (3,) and geometry.visibility.shape == (9,)
    assert geometry.quality_vector.shape == (6,) and len(geometry.quality_names) == 6
    assert sample.crop_sha256 and sample.prior_sha256


@package_required
def test_target_sample_contract_has_no_label_or_identity():
    dataset = CanonicalPackageDataset(PACKAGE, "target_test", CONFIG, mode="inference")
    sample = dataset[0]
    assert isinstance(sample, CanonicalTargetSample)
    assert not hasattr(sample, "label") and not hasattr(sample, "class_target")
    assert not hasattr(sample, "identity_embedding") and sample.identity_available is False
    assert not (FORBIDDEN_TARGET_FIELDS & set(vars(sample)))
    blob = json.dumps({k: str(v) for k, v in vars(sample).items() if k != "image" and k != "geometry"}).lower()
    for token in PRIVATE_TOKENS:
        assert token not in blob


@package_required
def test_priors_load_without_pickle_and_identity_only_on_train_live():
    from prism_fas.data.package.priors import load_prior
    dataset = CanonicalPackageDataset(PACKAGE, "source_train", CONFIG, mode="training")
    live = spoof = 0
    for position in range(0, len(dataset), 97):
        sample = dataset[position]
        arrays = load_prior(PACKAGE / dataset.index.rows[position]["prior_relative_path"])  # allow_pickle=False
        assert "parsing_labels" in arrays
        if sample.label == "live":
            live += 1; assert sample.identity_available and sample.identity_embedding.shape == (512,)
        else:
            spoof += 1; assert not sample.identity_available and sample.identity_embedding is None
    assert live and spoof


# --- collate ---------------------------------------------------------------

@package_required
def test_source_collate_shapes_and_identity_mask():
    dataset = CanonicalPackageDataset(PACKAGE, "source_train", CONFIG, mode="training")
    batch = collate_source_batch([dataset[i] for i in range(8)])
    assert batch["image"].shape == (8, 3, 224, 224) and batch["target"].shape == (8,)
    assert batch["bbox"].shape == (8, 4) and batch["landmarks"].shape == (8, 5, 2)
    assert batch["parsing"].shape == (8, 224, 224) and str(batch["parsing"].dtype) == "torch.int64"
    assert batch["pose"].shape == (8, 3) and batch["visibility"].shape == (8, 9) and batch["quality"].shape == (8, 6)
    assert batch["identity_embedding"].shape == (8, 512) and batch["identity_available"].shape == (8,)
    for position in range(8):
        if not bool(batch["identity_available"][position]):
            assert float(batch["identity_embedding"][position].abs().sum()) == 0.0   # masked placeholder


@package_required
def test_target_collate_isolation_and_rejection():
    dataset = CanonicalPackageDataset(PACKAGE, "target_test", CONFIG, mode="inference")
    batch = collate_target_batch([dataset[i] for i in range(4)])
    assert batch["image"].shape == (4, 3, 224, 224)
    assert "target" not in batch and "label" not in batch and "identity_embedding" not in batch
    assert not bool(batch["identity_available"].any())
    source = CanonicalPackageDataset(PACKAGE, "source_train", CONFIG, mode="training")
    with pytest.raises(TargetIsolationViolation):
        collate_target_batch([source[0]])
    with pytest.raises(TargetIsolationViolation):
        collate_source_batch([dataset[0]])


# --- loose / shard parity --------------------------------------------------

@package_required
@pytest.mark.parametrize("split,mode,want_label", [("source_train", "training", "live"), ("source_train", "training", "spoof"),
                                                   ("source_dev", "validation", None), ("target_test", "inference", None)])
def test_loose_shard_parity(split, mode, want_label):
    loose = CanonicalPackageDataset(PACKAGE, split, CONFIG, mode=mode)
    wanted = None
    for position in range(len(loose)):
        row = loose.index.rows[position]
        if want_label is None or row.get("label_live_spoof") == want_label:
            wanted = row["sample_id"]; break
    assert wanted is not None
    a = loose[loose.index_of(wanted)]
    b = next(sample for sample in CanonicalShardDataset(PACKAGE, split, CONFIG, mode=mode) if sample.sample_id == wanted)
    assert np.allclose(a.image, b.image, atol=1e-6)
    assert np.array_equal(a.geometry.parsing_labels, b.geometry.parsing_labels)
    assert np.array_equal(a.geometry.visibility, b.geometry.visibility)
    for name in ("bbox", "landmarks", "crop_box", "pose_ypr", "quality_vector"):
        assert np.allclose(getattr(a.geometry, name), getattr(b.geometry, name), atol=1e-6)
    assert a.dataset == b.dataset and a.project_split == b.project_split and a.crop_sha256 == b.crop_sha256
    assert a.identity_available == b.identity_available
    assert getattr(a, "label", None) == getattr(b, "label", None)
    if a.identity_available:
        assert np.allclose(a.identity_embedding, b.identity_embedding, atol=1e-6)


# --- sampler ---------------------------------------------------------------

@package_required
def test_sampler_exact_pool_quotas_and_no_duplicates():
    index = open_package(PACKAGE, "source_train", CONFIG, mode="training")
    sampler = BalancedDomainClassBatchSampler(index, CONFIG)
    assert sampler.steps_per_epoch == 45 and sampler.per_pool == 8
    for position, batch in enumerate(sampler):
        if position >= 6: break
        rows = [index.rows[i] for i in batch]
        counts = {}
        for row in rows: counts[(row["dataset"], row["label_live_spoof"])] = counts.get((row["dataset"], row["label_live_spoof"]), 0) + 1
        assert counts == {("casia_fasd", "live"): 8, ("casia_fasd", "spoof"): 8, ("msu_mfsd", "live"): 8, ("msu_mfsd", "spoof"): 8}
        ids = [row["sample_id"] for row in rows]
        assert len(set(ids)) == len(ids) == 32
        records = [row["source_record_id"] for row in rows]
        assert len(set(records)) == len(records)
        assert all(row["project_split"] == "source_train" for row in rows)


@package_required
def test_sampler_determinism_across_seed_and_epoch():
    index = open_package(PACKAGE, "source_train", CONFIG, mode="training")
    def run(config, epoch):
        sampler = BalancedDomainClassBatchSampler(index, config); sampler.set_epoch(epoch)
        return batch_fingerprint([[index.rows[i]["sample_id"] for i in batch] for n, batch in enumerate(sampler) if n < 5])
    assert run(CONFIG, 0) == run(CONFIG, 0)
    assert run(CONFIG, 0) != run(CONFIG, 1)
    other = CONFIG.model_copy(update={"sampler": CONFIG.sampler.model_copy(update={"seed": 999})})
    assert run(other, 0) != run(CONFIG, 0)


@package_required
def test_minority_pool_cycles_deterministically():
    index = open_package(PACKAGE, "source_train", CONFIG, mode="training")
    sampler = BalancedDomainClassBatchSampler(index, CONFIG)
    drawn = {"msu_mfsd/live": []}
    for batch in sampler:
        for position in batch:
            row = index.rows[position]
            if (row["dataset"], row["label_live_spoof"]) == ("msu_mfsd", "live"):
                drawn["msu_mfsd/live"].append(row["sample_id"])
    total = len(drawn["msu_mfsd/live"])
    assert total == 45 * 8                       # quota held despite a 120-sample pool
    assert len(set(drawn["msu_mfsd/live"])) <= 120
    assert len(set(drawn["msu_mfsd/live"])) > 1  # reuse, not a single repeated sample


def test_batch_size_must_divide_pool_count(tmp_path):
    index = type("Index", (), {"split": "source_train", "rows": tuple({"dataset": d, "label_live_spoof": l, "sample_id": f"{d}{l}{i}", "source_record_id": f"r{i}"} for d in ("casia_fasd", "msu_mfsd") for l in ("live", "spoof") for i in range(4)), "content_identity": "x" * 64})()
    bad = CONFIG.model_copy(update={"sampler": CONFIG.sampler.model_copy(update={"batch_size": 6})})
    with pytest.raises(SamplerConfigurationError, match="not divisible"):
        BalancedDomainClassBatchSampler(index, bad)


# --- failure modes ---------------------------------------------------------

@package_required
def test_missing_prior_and_incomplete_shard_are_rejected(tmp_path):
    import shutil
    from prism_fas.data.package.manifests import read_manifest
    root = tmp_path / "mini"
    (root / "manifests").mkdir(parents=True); (root / "images").mkdir(); (root / "priors").mkdir(); (root / "shards").mkdir()
    shutil.copy(PACKAGE / "PACKAGE_LOCK.json", root / "PACKAGE_LOCK.json")
    for name in ("source_train", "shards_index"):
        shutil.copy(PACKAGE / "manifests" / f"{name}.parquet", root / "manifests" / f"{name}.parquet")
    rows = read_manifest(root / "manifests" / "source_train.parquet")[:1]
    image = root / rows[0]["image_relative_path"]; image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PACKAGE / rows[0]["image_relative_path"], image)      # prior deliberately absent
    dataset = CanonicalPackageDataset(root, "source_train", CONFIG, mode="training")
    with pytest.raises(SampleContractError, match="prior missing"):
        dataset[dataset.index_of(rows[0]["sample_id"])]


@package_required
def test_incomplete_shard_triplet_is_rejected(tmp_path):
    import shutil
    root = tmp_path / "broken"; (root / "shards").mkdir(parents=True); (root / "manifests").mkdir()
    shutil.copy(PACKAGE / "PACKAGE_LOCK.json", root / "PACKAGE_LOCK.json")
    for name in ("source_train", "shards_index"):
        shutil.copy(PACKAGE / "manifests" / f"{name}.parquet", root / "manifests" / f"{name}.parquet")
    source = PACKAGE / "shards" / "source_train-00000.tar"
    with tarfile.open(source) as src, tarfile.open(root / "shards" / "source_train-00000.tar", "w") as dst:
        for info in src.getmembers()[:5]:
            if info.name.endswith(".npz"): continue          # drop one leg of the triplet
            dst.addfile(info, src.extractfile(info))
    with pytest.raises(SampleContractError):
        list(CanonicalShardDataset(root, "source_train", CONFIG, mode="training"))
