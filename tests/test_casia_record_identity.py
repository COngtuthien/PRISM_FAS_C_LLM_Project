from pathlib import Path

import pytest
import yaml

from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters.adapters import CasiaFasdAdapter
from prism_fas.data.preprocess_m2 import sample_id


def _definition():
    return DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[1] / "configs" / "data" / "casia_fasd.yaml").read_text()))


def test_same_video_stem_in_both_splits_yields_distinct_record_and_sample_identity(tmp_path):
    """Regression: train/ and test/ share s<subject>v<video> stems, which collapsed
    into one canonical video_id and produced conflicting duplicate sample_ids."""
    for split in ("train", "test"):
        directory = tmp_path / split / "live"
        directory.mkdir(parents=True)
        for frame in (1, 2):
            (directory / f"s10v1f{frame}.png").write_bytes(b"synthetic")

    records = CasiaFasdAdapter(tmp_path, _definition()).records()
    assert len(records) == 2
    video_ids = [r.video_id for r in records]
    assert len(set(video_ids)) == 2, video_ids
    assert {r.official_split for r in records} == {"train", "test"}
    assert all(r.video_id.startswith(r.official_split) for r in records)

    ids = {sample_id("casia_fasd", r.video_id, 0, r.adapter_version, "uniform-v1", "m2-v1") for r in records}
    assert len(ids) == 2


def test_duplicate_canonical_video_ids_are_rejected(tmp_path, monkeypatch):
    directory = tmp_path / "train" / "live"
    directory.mkdir(parents=True)
    (directory / "s10v1f1.png").write_bytes(b"synthetic")
    (tmp_path / "test" / "live").mkdir(parents=True)
    (tmp_path / "test" / "live" / "s10v1f1.png").write_bytes(b"synthetic")
    definition = _definition().model_copy(update={"splits": {"train": "same", "test": "same"}})
    with pytest.raises(ValueError, match="Duplicate CASIA video id"):
        CasiaFasdAdapter(tmp_path, definition).records()
