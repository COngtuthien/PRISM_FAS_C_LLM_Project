# E8 QMATCH Membership Summary (Official, Frozen-F1)

**This is the official E8 frozen-F1 QMATCH membership. It is NOT an E7 rescue bank.**

E7 remains scientifically closed: EXT-F1/F2/F3 all `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` -- **0/3 E7 core matched synthetic banks.**

**Selector implementation commit:** `1bc21807564650141d22b401003d91c6b10d7b32`
**Input-binding rule identity:** `8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8`
**Selector rule identity:** `95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126`
**Input q-table SHA256:** `87fdc8ea594a487bfef5206c5a0f0c763c7d19451a5f915e67e2237fd7431cd9`
**Input population:** `FROZEN_F1_C6_NOMINAL` (profile `NOMINAL`)

## Route x q-bin count table

### Physics

| q-bin | RND | DET | LLM | n_b | limiting arm(s) | selected RND/DET/LLM |
|---|---|---|---|---|---|---|
| [0.0,0.1) | 0 | 2 | 0 | **0** | LLM, RND | 0 / 0 / 0 |
| [0.1,0.2) | 0 | 0 | 0 | **0** | DET, LLM, RND | 0 / 0 / 0 |
| [0.2,0.3) | 0 | 0 | 4 | **0** | DET, RND | 0 / 0 / 0 |
| [0.3,0.4) | 3 | 4 | 11 | **3** | RND | 3 / 3 / 3 |
| [0.4,0.5) | 24 | 12 | 23 | **12** | DET | 12 / 12 / 12 |
| [0.5,0.6) | 30 | 38 | 74 | **30** | RND | 30 / 30 / 30 |
| [0.6,0.7) | 80 | 72 | 134 | **72** | DET | 72 / 72 / 72 |
| [0.7,0.8) | 161 | 136 | 165 | **136** | DET | 136 / 136 / 136 |
| [0.8,0.9) | 168 | 190 | 98 | **98** | LLM | 98 / 98 / 98 |
| [0.9,1.0] | 46 | 58 | 3 | **3** | LLM | 3 / 3 / 3 |

### GPAT

| q-bin | RND | DET | LLM | n_b | limiting arm(s) | selected RND/DET/LLM |
|---|---|---|---|---|---|---|
| [0.0,0.1) | 0 | 0 | 0 | **0** | DET, LLM, RND | 0 / 0 / 0 |
| [0.1,0.2) | 0 | 0 | 1 | **0** | DET, RND | 0 / 0 / 0 |
| [0.2,0.3) | 0 | 0 | 2 | **0** | DET, RND | 0 / 0 / 0 |
| [0.3,0.4) | 2 | 4 | 7 | **2** | RND | 2 / 2 / 2 |
| [0.4,0.5) | 17 | 19 | 22 | **17** | RND | 17 / 17 / 17 |
| [0.5,0.6) | 43 | 33 | 43 | **33** | DET | 33 / 33 / 33 |
| [0.6,0.7) | 95 | 115 | 107 | **95** | RND | 95 / 95 / 95 |
| [0.7,0.8) | 212 | 178 | 182 | **178** | DET | 178 / 178 / 178 |
| [0.8,0.9) | 134 | 152 | 143 | **134** | RND | 134 / 134 / 134 |
| [0.9,1.0] | 9 | 11 | 5 | **5** | LLM | 5 / 5 / 5 |

## Totals

| Metric | Value |
|---|---|
| Physics total (per arm) | **354** |
| GPAT total (per arm) | **464** |
| Total per arm | **818** |
| Grand total (all arms) | **2454** |
| Empty bins | gpat:bin0, gpat:bin1, gpat:bin2, physics:bin0, physics:bin1, physics:bin2 |
| q min / max | 0.085553 / 0.941574 |
| q == 1.0 count | 0 |
| Source-domain composition | not available at per-sample/per-bin granularity (no source-domain column in the reconstructed parquet); whole-bank aggregate is casia_fasd + msu_mfsd only |

## Integrity

- Target firewall: **PASS** -- `target_access=false`, `target_labels_accessed=false`, no `siw_mv2` sample present in the frozen input population.
- LLM API calls: **0**
- GPU status: **not used**
- Detector training: **not started**
