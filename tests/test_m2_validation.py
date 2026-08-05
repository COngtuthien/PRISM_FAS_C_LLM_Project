from pathlib import Path
from prism_fas.config.models import load_paths
from prism_fas.data.preprocess_m2 import load_m2_config
from prism_fas.data.m2_validation import validate_m2, status_m2

ROOT=Path(__file__).parents[1]
def test_actual_small_acceptance_validation_passes():
    paths=load_paths(ROOT/'configs/paths.local.yaml'); result=validate_m2(paths,load_m2_config(ROOT/'configs/data/preprocess_m2.yaml'))
    assert result['passed'] and not result['errors']
def test_status_reports_expected_counts():
    paths=load_paths(ROOT/'configs/paths.local.yaml'); result=status_m2(paths,load_m2_config(ROOT/'configs/data/preprocess_m2.yaml'))
    assert result['completed_samples']==36 and result['failed_samples']==0 and result['target_isolation']
