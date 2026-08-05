from pathlib import Path
import pytest, yaml
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters import MsuMfsdAdapter
def definition() -> DatasetDefinition: return DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[2]/"configs/data/msu_mfsd.yaml").read_text()))
def fixture(root: Path) -> None:
    base=root/"MSU-MFSD-Publish.zip"; (base/"train_sub_list.txt").parent.mkdir(parents=True); (base/"train_sub_list.txt").write_text("01\n"); (base/"test_sub_list.txt").write_text("02\n")
    for rel in ["scene01/real/real_client001_android_SD_scene01.mp4", "scene01/attack/attack_client002_laptop_SD_ipad_video_scene01.mov"]:
        p=base/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b"x")
def test_msu_valid_and_deterministic(tmp_path: Path):
    fixture(tmp_path); adapter=MsuMfsdAdapter(tmp_path,definition()); records=adapter.records(); assert len(records)==2 and records == adapter.records()
def test_msu_invalid_filename(tmp_path: Path):
    fixture(tmp_path); p=tmp_path/"MSU-MFSD-Publish.zip/scene01/real/nope.mp4"; p.write_bytes(b"x")
    with pytest.raises(ValueError,match="exactly one"): MsuMfsdAdapter(tmp_path,definition()).records()
def test_msu_split_overlap(tmp_path: Path):
    fixture(tmp_path); (tmp_path/"MSU-MFSD-Publish.zip/test_sub_list.txt").write_text("01\n")
    with pytest.raises(ValueError,match="split overlap"): MsuMfsdAdapter(tmp_path,definition()).records()
