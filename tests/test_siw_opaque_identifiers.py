from pathlib import Path

import pytest
import yaml

from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters.adapters import SiWMv2Adapter, opaque_record_id
from prism_fas.data.preprocess_m2 import sample_id

PRIVATE_TOKENS = ("live", "spoof", "attack", "taxonomy", "subject", "session", "replay", "mask", "paper", "makeup", "477", ".mov", ".avi", ".mp4")


def _definition():
    return DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[1] / "configs" / "data" / "siw_mv2.yaml").read_text()))


def _dataset(root, names):
    for relative in names:
        p = root / "SiW-Mv2" / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic-video")
    return root


@pytest.mark.parametrize("relative", ["Live/Live_477.mov", "Spoof/Spoof_477.mov"])
def test_opaque_id_hides_class_and_filename(tmp_path, relative):
    records = SiWMv2Adapter(_dataset(tmp_path, [relative]), _definition()).inference_records()
    assert len(records) == 1
    identifier = records[0].video_id
    assert identifier.startswith("siw_") and len(identifier) == 20
    lowered = identifier.lower()
    for token in PRIVATE_TOKENS:
        assert token not in lowered, f"{token!r} leaked into {identifier!r}"
    assert Path(relative).stem.lower() not in lowered and "/" not in identifier
    sid = sample_id("siw_mv2", identifier, 0, records[0].adapter_version, "uniform-v1", "m2-v1")
    for token in PRIVATE_TOKENS:
        assert token not in sid.lower()


def test_same_relative_path_is_stable_across_dataset_roots(tmp_path):
    definition = _definition()
    a = SiWMv2Adapter(_dataset(tmp_path / "root_a", ["Live/Live_477.mov"]), definition).inference_records()[0]
    b = SiWMv2Adapter(_dataset(tmp_path / "elsewhere" / "root_b", ["Live/Live_477.mov"]), definition).inference_records()[0]
    assert a.video_id == b.video_id == opaque_record_id("siw_mv2", "Live/Live_477.mov")


def test_distinct_relative_paths_yield_distinct_ids(tmp_path):
    records = SiWMv2Adapter(_dataset(tmp_path, ["Live/Live_1.mov", "Live/Live_2.mov", "Spoof/Spoof_1.mov"]), _definition()).inference_records()
    assert len({r.video_id for r in records}) == len(records) == 3


def test_opaque_collision_is_rejected(tmp_path, monkeypatch):
    import prism_fas.data.adapters.adapters as adapters
    monkeypatch.setattr(adapters, "opaque_record_id", lambda dataset, relative: "siw_collision")
    with pytest.raises(ValueError, match="Duplicate SiW-Mv2 opaque record id"):
        SiWMv2Adapter(_dataset(tmp_path, ["Live/Live_1.mov", "Live/Live_2.mov"]), _definition()).inference_records()


def test_private_evaluation_records_share_the_same_opaque_id(tmp_path):
    root = _dataset(tmp_path, ["Live/Live_477.mov"])
    adapter = SiWMv2Adapter(root, _definition())
    inference, private = adapter.inference_records()[0], adapter.private_evaluation_records()[0]
    assert inference.video_id == private.video_id
    assert private.label == "live" and "live" not in private.video_id.lower()
