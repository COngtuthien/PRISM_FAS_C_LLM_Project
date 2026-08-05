from pathlib import Path
import yaml
import pytest
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters import CasiaFasdAdapter, SiWMv2Adapter
def definition(name: str) -> DatasetDefinition:
    return DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[2] / "configs" / "data" / f"{name}.yaml").read_text()))
def test_casia_is_deterministic(tmp_path: Path):
    for name in ["s1v2f0.png", "s1v2f1.png"]:
        p=tmp_path / "train" / "live" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    adapter=CasiaFasdAdapter(tmp_path, definition("casia_fasd")); assert adapter.records() == adapter.records()
def test_siw_hides_private_label(tmp_path: Path):
    p=tmp_path / "SiW-Mv2" / "Live" / "Live_1.avi"; p.parent.mkdir(parents=True); p.write_bytes(b"x")
    record=SiWMv2Adapter(tmp_path, definition("siw_mv2")).inference_records()[0]
    assert "label" not in record.model_dump()
def test_invalid_layout_fails(tmp_path: Path):
    p=tmp_path / "train" / "live" / "bad.png"; p.parent.mkdir(parents=True); p.write_bytes(b"x")
    with pytest.raises(ValueError): CasiaFasdAdapter(tmp_path, definition("casia_fasd")).records()
