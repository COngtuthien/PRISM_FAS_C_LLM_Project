# M2 design

## M2B1a manifests

M2A JSONL is migrated through strict source/target/failure schemas to atomic,
sorted Parquet. Target manifests undergo recursive prohibited-field isolation
validation before writing.

Inputs are M1 canonical records only. `video_file`, `image_sequence`, and `single_image` are dispatched by explicit media typing. Uniform numeric indices are deterministic, sorted and deduplicated; sample IDs hash stable record fields only and never absolute paths or target labels.

Frames/crops and state live below `data/work/m2`. OpenCV is the primary video backend; image sequences use natural numeric ordering. Frame decoding failures become failure records and do not stop later records. State/manifests are atomically written; an existing matching output hash is skipped on resume.

`FaceDetector` has mock and SCRFD implementations. SCRFD receives its local configured ONNX model, uses CPU by default, and returns score, bbox and five landmarks. `largest_valid_face` filters threshold/minimum size then sorts by area, score and coordinates. Crops apply explicit padding, clamp bounds, resize to the configured square and retain transform metadata.

Source manifests contain subject, split and label; target schemas intentionally omit all labels, taxonomy and private metadata. Tests cover deterministic sampling/IDs, dispatch, selection/crops, failure/resume and target serialization. Completion additionally requires an actual local SCRFD small run on all three datasets.
