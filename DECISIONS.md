# Decisions

- Python 3.11 is the project baseline (`requires-python >=3.11`). The host exposes Python 3.13 only; no global runtime was changed.
- MSU-MFSD is blocked for record mapping until explicit official protocol/layout YAML is supplied.
- SiW-Mv2 inference records intentionally omit labels; the private adapter is evaluator-only.
- M2 defaults (4 uniform samples, 5% endpoint exclusion, 0.5 SCRFD threshold, 25% padding, 224px crops) are implementation defaults, not target-tuned values.
- SCRFD input is frozen at 320 under `scrfd_source_policy_v1`: source-only CASIA/MSU validation found equal valid-detection rates for 256 and 320, so the approved rule selects 320. Threshold remains 0.5.
