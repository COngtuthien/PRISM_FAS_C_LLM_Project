from pathlib import Path
import yaml
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.audit.audit import audit_dataset
def test_missing_metadata_is_reported(tmp_path: Path):
    definition=DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[2] / "configs" / "data" / "msu_mfsd.yaml").read_text()))
    report=audit_dataset(definition, tmp_path)
    assert report["errors"] and report["record_count"] == 0
