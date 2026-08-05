from pathlib import Path
import numpy as np
from prism_fas.data.preprocess_m2 import Detection,MockFaceDetector,crop_face,media_type,natural_paths,sample_id,select_largest,uniform_indices
def test_uniform_deterministic_no_duplicates(): assert uniform_indices(3,9)==[0,1,2] and uniform_indices(10,4)==uniform_indices(10,4)
def test_natural_ordering(tmp_path:Path):
    paths=[tmp_path/f"frame_{n}.png" for n in [10,2,1]]; assert [p.name for p in natural_paths(paths)]==["frame_1.png","frame_2.png","frame_10.png"]
def test_stable_id_and_collision_inputs():
    assert sample_id("d","v",1,"a","s","p")==sample_id("d","v",1,"a","s","p")
    assert sample_id("d","v",1,"a","s","p")!=sample_id("d","v",2,"a","s","p")
def test_dispatch_selection_and_crop():
    assert media_type(Path("x.mp4"))=="video_file" and media_type(Path("x.png"))=="single_image"
    selected=select_largest([Detection(bbox=(1,1,5,5),score=.9),Detection(bbox=(2,2,12,12),score=.6)],.5,2); assert selected is not None and selected.bbox==(2,2,12,12)
    crop,box=crop_face(np.zeros((20,20,3),np.uint8),selected,.25,16); assert crop.shape==(16,16,3) and box==(0,0,14,14)
def test_no_face(): assert select_largest([],0.5,16) is None
