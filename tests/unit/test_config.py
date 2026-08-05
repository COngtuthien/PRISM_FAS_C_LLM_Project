from pathlib import Path
import pytest
from pydantic import ValidationError
from prism_fas.config.models import PathsConfig, load_paths
def test_unknown_key_rejected():
    with pytest.raises(ValidationError): PathsConfig.model_validate({"workspace_root":"x","project_root":"x","raw_datasets":{"casia_fasd":"x","msu_mfsd":"x","siw_mv2":"x"},"model_cache":"x","work_root":"x","processed_root":"x","package_root":"x","runs_root":"x","reports_root":"x","nope":1})
def test_missing_config_path():
    with pytest.raises(FileNotFoundError): load_paths(Path("does-not-exist.yaml"))
