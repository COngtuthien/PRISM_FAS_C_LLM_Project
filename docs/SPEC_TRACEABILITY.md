# Spec traceability

| Requirement | Milestone | Module/files | Test | Status |
| M2B1a strict manifests and target isolation | M2 | `data/manifests`, migration reports | `test_manifests_m2b1a.py` | implemented/tested |
| M2B1b completed index and resume safety | M2 | `data/manifests/resume.py` | `test_resume_m2b1b.py` | implemented/tested |
|---|---|---|---|---|
| Spec ingestion/source provenance | M0 | `docs/SPEC_SOURCE.json`, snapshot | source inspection | implemented |
| Strict paths/runtime/repro config | M0 | `config/models.py` | `test_config.py` | implemented/tested |
| Hashes, atomic writes, metadata | M0 | `utils/core.py` | `test_core.py` | implemented/tested |
| Typer CLI and resolved config | M0 | `cli/main.py` | `test_cli.py` | implemented/tested |
| Canonical video record/adapters | M1 | `data/adapters/` | `test_adapters.py` | implemented/tested |
| Explicit YAML layout rules | M1 | `configs/data/*.yaml` | adapter tests | implemented/tested |
| Raw layout audit/reports | M1 | `data/audit/audit.py` | `test_audit.py` | implemented/tested |
| Target-label isolation | M1 | `SiWMv2Adapter` | leakage test | implemented/tested |
| Actual MSU record mapping | M1 | `configs/data/msu_mfsd.yaml`, `MsuMfsdAdapter` | `test_msu_adapter.py`, raw audit | implemented/tested |
| M2 deterministic sampling/media/crop primitives | M2 | `data/preprocess_m2.py` | `test_preprocess_m2.py` | implemented/tested |
| M2 end-to-end manifests, resume and three-dataset crops | M2 | preprocessing runner | N/A | blocked |
