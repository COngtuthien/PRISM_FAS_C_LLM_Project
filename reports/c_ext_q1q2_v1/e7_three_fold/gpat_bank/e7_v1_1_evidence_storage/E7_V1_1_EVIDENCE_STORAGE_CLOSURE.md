# E7-v1.1 Evidence Storage Closure

**Classification:** `COMPLETE_E7_V1_1_EVIDENCE_STORAGE_CLOSURE`
**Scope:** storage / provenance only. No experiment was run, no candidate was rendered, no GPAT/detector was trained, no LLM was called, and no historical raw evidence file was modified to produce this document.

**Scientific stage:** E7-v1.1 three-fold reserve
**Scientific closure commit:** `bf4d3ee6d1c95f380c225033899f7f75e9ebd88d`
**Reserve schedule rule identity:** `7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`

## A. Frozen scientific status (referenced, not altered)

This storage task does **not** reopen E7. Referenced from `../e7_v1_1_reserve/E7_V1_1_THREE_FOLD_SCIENTIFIC_CLOSURE.json` (commit `bf4d3ee6d1c95f380c225033899f7f75e9ebd88d`):

| Fold | Status |
|---|---|
| EXT-F1 | `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` |
| EXT-F2 | `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` |
| EXT-F3 | `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` |

**0 / 3 core matched synthetic banks were produced.**

## B. Raw evidence snapshot

- Directory: `gpu_evidence/e7_v1_1_reserve/`
- File count: **90** (JSON = 61, SHA256 = 17, LOG = 12)
- Total raw bytes: **246968715**
- Symlink count: **0**
- Laptop/GPU per-file SHA256 comparison (independent hashing under `LC_ALL=C`): `ONLY_GPU=none`, `ONLY_LAPTOP=none`, `HASH_MISMATCH=none`, **mismatch_count = 0** -- laptop and GPU raw evidence snapshots are byte-identical file-for-file.
- Master evidence manifest `E7_V1_1_THREE_FOLD_TERMINAL_EVIDENCE.sha256`: already verified successfully; **not rewritten** by this task.
- Pre-write full manifest check (`E7_V1_1_FULL_RAW_EVIDENCE.sha256`): **90/90 OK**, 0 failed.
- Post-write full manifest check (same manifest, re-run after creating this closure): **90/90 OK**, 0 failed -- proves this reporting/storage task did not mutate historical evidence.

## C. Console archive

- File: `E7_V1_1_CONSOLE_LOGS.tar.gz` (this directory)
- Contents: exactly the 12 raw console logs
- Size: **9409820 bytes**
- SHA256: `0f17e4a4c64509fc21ae4363512b16f0eb67443cdc8ab6745b64cc4096d9c690`
- Restore verification: extracted to a temporary directory **outside the repository**; `sha256sum -c` against `E7_V1_1_CONSOLE_LOGS_RAW.sha256` -> **12/12 OK, restore RC = 0**. Temporary extraction directory deleted afterward; the raw evidence directory itself was never touched.
- GPU archive verification: archive copied to the GPU storage directory; `sha256sum -c E7_V1_1_CONSOLE_LOGS_ARCHIVE.sha256` there returned **OK**.
- **Deterministic-mtime warning (not an integrity failure):** the archive's fixed deterministic mtime (`2026-09-07 00:00:00 UTC`) was slightly later than the verifier's wall-clock time at the moment of an immediate post-creation extraction, so `tar` printed a "timestamp ... is N s in the future" warning per file. Extraction RC was 0 and all 12 extracted files' SHA256 matched the raw-log manifest. This fact is recorded so the warning is never later misclassified as corruption.

**The Git archive is a lossless storage representation of the 12 original console logs**, validated by extraction followed by SHA256 verification against the raw-log manifest. The archived logs are scientific provenance only; compression does not alter the historical scientific observations.

## D. Preservation state

| Item | Laptop | GPU |
|---|---|---|
| Raw uncompressed `.log` files | preserved | preserved |
| Compressed archive | preserved | preserved |

Compressed archive committed to Git: **true, after this commit succeeds.** Raw uncompressed logs are **not** deleted by this task and are **not** committed to Git (superseded, for Git purposes, by the lossless archive).

## E. Storage policy -- final decision

**Git LFS used: false.** Reason: the compressed archive is approximately 9 MB and does not warrant LFS.

Committed to Git by this closure:
1. all 61 raw evidence JSON artifacts from `gpu_evidence/e7_v1_1_reserve/`;
2. all 17 existing raw evidence SHA256 manifests from that directory;
3. `E7_V1_1_CONSOLE_LOGS.tar.gz`;
4. `E7_V1_1_CONSOLE_LOGS_RAW.sha256`;
5. `E7_V1_1_CONSOLE_LOGS_ARCHIVE.sha256`;
6. `E7_V1_1_FULL_RAW_EVIDENCE.sha256`;
7. this closure's `.md` and `.json` files.

**Not** committed: the 12 individual raw `.log` files. The full console process remains recoverable from Git via the archive, without putting ~233 MB of duplicate uncompressed text logs into Git history.

## F. Downstream status (unaffected by this task)

`E8` remains `READY` per `E7_V1_1_THREE_FOLD_SCIENTIFIC_CLOSURE.json`'s `downstream_dependency_classification`; it is **not** started by this task. No GPU scientific work, candidate generation, detector training, or LLM call was performed to produce this closure.
