# Project status

- Current milestone: **M2 COMPLETED**. M3 is NOT STARTED.
- M0: COMPLETED; M1: COMPLETED; M2: COMPLETED; M3: NOT STARTED.
- Official M2 dataset namespace: `full_preprocessing_v2`, written under
  `<work_root>/m2/<preprocessing_version>/<preprocessing_config_hash>/full_preprocessing_v2`.
  The earlier `full_preprocessing` namespace is retained locally as an audit artifact and must not be modified.

## Final M2 execution (full_preprocessing_v2)

| Dataset | Role | Records | Selected | Successful | Failed |
|---|---|---|---|---|---|
| casia_fasd | source | 600 | 2400 | 2399 | 1 |
| msu_mfsd | source | 280 | 1120 | 1120 | 0 |
| siw_mv2 | target | 785 | 3140 | 3140 | 0 |
| **Total** | | **1665** | **6660** | **6659** | **1** |

Manifest counts: `source_frames 3519`, `source_crops 3519`, `target_frames 3140`,
`target_crops 3140`, `preprocessing_failures 1`; 6659 crop files on disk.

Reconciliation: `selected = successful + failed` and `successful = source_crops + target_crops`.

The single failure is a genuine detector outcome (`casia_fasd`, `error_code=no_face`,
`stage=detector`, `recoverable=true`), routed to the failure manifest with the run continuing.

## Validation

- Full-profile validation: **passed = true**, 0 errors, 35/35 checks.
- Missing crops 0, orphan crops 0, SHA mismatches 0, duplicate sample IDs 0,
  unreadable crops 0, wrong dimensions 0, temporary artifacts 0.
- Target privacy/isolation: 0 matches.
- Tests: `266 passed, 0 failed, 0 skipped` via `python -m pytest -q`.

## Target isolation

SiW-Mv2 target identifiers are opaque `siw_<16 hex>` values derived deterministically from the
dataset-root-relative path, so no class, subject, session or raw filename information reaches the
target manifests. Target rows carry no label, attack, taxonomy, subject or session field.

## Data policy

Generated data is intentionally **not** committed to GitHub: raw datasets, model weights,
crop images, Parquet manifests, run logs and local path configuration are ignored. The dataset is
reproducible from source using the documented CLI and the frozen preprocessing configuration.

- Blockers: none for M2.
- Next milestone: M3 (not started).
- Continue command: `python -m pytest -q`
