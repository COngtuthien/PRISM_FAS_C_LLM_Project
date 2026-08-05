# Decisions

- Python 3.11 is the project baseline (`requires-python >=3.11`). The host exposes Python 3.13 only; no global runtime was changed.
- MSU-MFSD is blocked for record mapping until explicit official protocol/layout YAML is supplied.
- SiW-Mv2 inference records intentionally omit labels; the private adapter is evaluator-only.
- M2 defaults (4 uniform samples, 5% endpoint exclusion, 0.5 SCRFD threshold, 25% padding, 224px crops) are implementation defaults, not target-tuned values.
- SCRFD input is frozen at 320 under `scrfd_source_policy_v1`: source-only CASIA/MSU validation found equal valid-detection rates for 256 and 320, so the approved rule selects 320. Threshold remains 0.5.
- M3A packages preserve the exact M2 JPEG crop bytes instead of re-encoding to PNG. The spec snapshot suggested PNG, but the frozen M2 checkpoint produces validated JPEG artifacts whose SHA-256 identity is recorded in the M2 manifests; re-encoding would break that identity and apply a second lossy image transformation. Package rows therefore record `image_format="jpeg"` with the `.jpg` extension, and packaging verifies the copied bytes still match the M2 crop SHA-256.
- M3A contrast is reported as `contrast_michelson` rather than RMS contrast: for a single normalized grayscale crop, RMS contrast is numerically identical to `brightness_std`, so Michelson contrast is used to add independent information. The formula is documented in `configs/data/package_m3a.yaml`.
- M3A computes only detector-geometry and deterministic image-quality priors. Parsing masks, pose, visibility and identity embeddings require pretrained models and are recorded as `*_status="not_computed"`; no zero-filled placeholder arrays are written.
