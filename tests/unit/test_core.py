from pathlib import Path
from prism_fas.utils.core import atomic_json_write, atomic_yaml_write, stable_json_hash
def test_stable_hash(): assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash({"a": 1, "b": 2})
def test_atomic_writes(tmp_path: Path):
    atomic_json_write(tmp_path / "x.json", {"x": 1}); atomic_yaml_write(tmp_path / "x.yaml", {"x": 1})
    assert (tmp_path / "x.json").read_text().startswith("{") and "x: 1" in (tmp_path / "x.yaml").read_text()
