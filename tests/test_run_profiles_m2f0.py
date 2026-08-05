from pathlib import Path
import pytest
from prism_fas.data.run_profiles import load_profiles, profile_root, PreprocessingRunProfile

def test_profiles_are_strict_and_separate(tmp_path):
    p=Path(__file__).parents[1]/'configs/data/m2_run_profiles.yaml'; profiles=load_profiles(p)
    assert profiles['small_acceptance'].default_record_limit==3
    assert profile_root(tmp_path,'v','h',profiles['small_acceptance']) != profile_root(tmp_path,'v','h',profiles['full_preprocessing'])
    with pytest.raises(Exception): PreprocessingRunProfile.model_validate({**profiles['small_acceptance'].model_dump(),'unexpected':True})
